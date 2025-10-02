"""SNS 업로드 환경을 모사하는 이미지 증강 모듈.

여러 SNS 플랫폼은 업로드된 이미지를 반복적으로 리사이즈하거나 재인코딩하며,
톤 조정이나 스크린샷 공유 과정에서 다양한 왜곡이 발생한다. 이 모듈은 이러한
분포를 학습 단계에서 미리 경험하게 하여 실서비스 환경과의 격차를 줄인다.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

from core.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class AugConfig:
    """증강 강도 범위를 정의하는 설정 데이터 클래스."""

    jpeg_q: Tuple[int, int] = (35, 90)
    webp_q: Tuple[int, int] = (35, 90)
    resize_long: Tuple[int, int] = (512, 2048)
    aspect_ratio: Tuple[float, float] = (0.6, 1.8)
    noise_sigma: Tuple[float, float] = (0.0, 8.0)
    gamma: Tuple[float, float] = (0.8, 1.2)
    contrast: Tuple[float, float] = (0.8, 1.2)
    saturation: Tuple[float, float] = (0.8, 1.2)
    brightness: Tuple[float, float] = (0.85, 1.15)
    repeats: Tuple[int, int] = (1, 3)
    watermark_alpha: Tuple[float, float] = (0.3, 0.6)


def _random_long_short(size: Tuple[int, int], cfg: AugConfig) -> Tuple[int, int]:
    """긴 변과 비율 범위를 토대로 새로운 크기를 샘플링한다."""

    long_min, long_max = cfg.resize_long
    aspect_min, aspect_max = cfg.aspect_ratio
    long_side = random.randint(long_min, long_max)
    aspect = random.uniform(aspect_min, aspect_max)
    short_side = int(long_side / aspect)
    if size[0] >= size[1]:
        return long_side, short_side
    return short_side, long_side


def _resize_variant(image: Image.Image, cfg: AugConfig) -> Tuple[Image.Image, str]:
    """SNS 리사이즈 정책을 흉내 내어 이미지를 변형한다."""

    w, h = image.size
    new_w, new_h = _random_long_short((w, h), cfg)
    mode = random.choice(["center", "cover", "letterbox"])
    if mode == "center":
        resized = image.resize((new_w, new_h), resample=random.choice([
            Image.BILINEAR, Image.BICUBIC, Image.LANCZOS
        ]))
        return resized, f"resize_center_{new_w}x{new_h}"
    if mode == "cover":
        resized = ImageOps.fit(
            image,
            (new_w, new_h),
            method=random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]),
            bleed=0.0,
            centering=(0.5, 0.5),
        )
        return resized, f"resize_cover_{new_w}x{new_h}"
    # letterbox 모드
    base = Image.new(image.mode, (new_w, new_h), color=(0, 0, 0))
    resized = ImageOps.contain(
        image,
        (new_w, new_h),
        method=random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]),
    )
    paste_x = (new_w - resized.size[0]) // 2
    paste_y = (new_h - resized.size[1]) // 2
    base.paste(resized, (paste_x, paste_y))
    return base, f"resize_letterbox_{new_w}x{new_h}"


def _reencode(image: Image.Image, cfg: AugConfig) -> Tuple[Image.Image, str]:
    """JPEG/WebP 재인코딩을 수행하여 손실 압축 흔적을 만든다."""

    fmt = random.choice(["JPEG", "WEBP"])
    quality_range = cfg.jpeg_q if fmt == "JPEG" else cfg.webp_q
    quality = random.randint(*quality_range)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, quality=quality, optimize=True)
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    return decoded, f"reencode_{fmt.lower()}_q{quality}"


def _resample_chain(image: Image.Image) -> Tuple[Image.Image, str]:
    """업/다운 샘플링을 여러 번 적용해 라인 아티팩트를 만든다."""

    resamplers = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    steps = random.randint(1, 2)
    h, w = image.height, image.width
    ops: List[str] = []
    current = image
    for _ in range(steps):
        scale = random.uniform(0.5, 1.5)
        interp = random.choice(resamplers)
        new_size = (max(32, int(w * scale)), max(32, int(h * scale)))
        current = current.resize(new_size, resample=interp)
        ops.append(f"resize_{new_size[0]}x{new_size[1]}_{interp}")
    current = current.resize((w, h), resample=random.choice(resamplers))
    ops.append("restore_original")
    return current, ",".join(ops)


def _apply_noise_and_blur(image: Image.Image, cfg: AugConfig) -> Tuple[Image.Image, str]:
    """노이즈와 블러를 추가하여 촬영/압축 노이즈를 흉내낸다.

    - 가우시안 노이즈를 추가하고, 확률적으로 가우시안 블러를 적용한다.
    - 별도로 평균/가우시안 커널 기반의 정사각형 커널 블러를 적용한다.
      커널 크기는 3/5/7 중 랜덤 선택하며, ImageFilter.Kernel의 요건에 맞게
      (k, k) 크기와 길이 k*k의 커널 리스트를 전달한다.
    """

    ops: List[str] = []
    # 1) 노이즈 추가
    sigma = random.uniform(*cfg.noise_sigma)
    arr = np.array(image).astype(np.float32)
    if sigma > 0:
        noise = np.random.normal(0.0, sigma, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        ops.append(f"gaussian_noise_{sigma:.2f}")
    result = Image.fromarray(arr.astype(np.uint8)).convert("RGB")

    # 2) 가우시안 블러(반지름 기반)
    if random.random() < 0.5:
        radius = random.uniform(1.0, 3.0)
        result = result.filter(ImageFilter.GaussianBlur(radius=radius))
        ops.append(f"gaussian_blur_{radius:.1f}")

    # 3) 커널 블러(정사각형, 홀수 크기)
    if random.random() < 0.3:
        k = random.choice([3, 5, 7])
        if random.random() < 0.5:
            # 평균 블러: 모든 원소 1/(k*k)
            kernel = np.ones((k, k), dtype=np.float32) / float(k * k)
            kind = "avg"
        else:
            # 간단 가우시안 커널 생성 후 정규화
            x = np.linspace(-1.0, 1.0, k)
            xv, yv = np.meshgrid(x, x)
            sigma_k = 0.5
            g = np.exp(-(xv**2 + yv**2) / (2.0 * sigma_k**2))
            kernel = (g / g.sum()).astype(np.float32)
            kind = "gauss"
        kernel_list = kernel.reshape(-1).tolist()
        # scale은 커널 합으로 설정(이미 1.0에 가깝지만 안전하게 합 사용)
        result = result.filter(ImageFilter.Kernel(size=(k, k), kernel=kernel_list, scale=float(sum(kernel_list))))
        ops.append(f"kernel_blur_{kind}_{k}x{k}")

    return result, ",".join(ops)


def _apply_tone(image: Image.Image, cfg: AugConfig) -> Tuple[Image.Image, str]:
    """톤/색상 조정을 적용한다."""

    gamma = random.uniform(*cfg.gamma)
    contrast = random.uniform(*cfg.contrast)
    saturation = random.uniform(*cfg.saturation)
    brightness = random.uniform(*cfg.brightness)

    arr = np.array(image).astype(np.float32) / 255.0
    arr = np.power(arr, gamma)
    arr = np.clip(arr * brightness, 0.0, 1.0)
    toned = Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")

    toned = ImageEnhance.Contrast(toned).enhance(contrast)
    toned = ImageEnhance.Color(toned).enhance(saturation)
    return toned, f"tone_gamma{gamma:.2f}_ctr{contrast:.2f}_sat{saturation:.2f}_bri{brightness:.2f}"


def _add_watermark(image: Image.Image, cfg: AugConfig) -> Tuple[Image.Image, str]:
    """작은 워터마크/스티커를 합성하여 SNS 공유 상황을 모사한다."""

    if random.random() > 0.1:
        return image, "watermark_skip"

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    w, h = image.size
    box_w = int(w * random.uniform(0.1, 0.2))
    box_h = int(h * random.uniform(0.05, 0.12))
    x = random.randint(0, w - box_w)
    y = random.randint(0, h - box_h)
    alpha = int(random.uniform(*cfg.watermark_alpha) * 255)
    color = (255, 255, 255, alpha)
    draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=8, fill=color)
    blended = Image.alpha_composite(image.convert("RGBA"), overlay)
    return blended.convert("RGB"), f"watermark_{box_w}x{box_h}_alpha{alpha}"


def _screenshot_simulation(image: Image.Image) -> Tuple[Image.Image, str]:
    """스크린샷 공유를 흉내 내기 위해 여백과 약한 블러를 적용한다."""

    border = random.randint(8, 24)
    bg_color = tuple(random.randint(200, 255) for _ in range(3))
    canvas = Image.new("RGB", (image.width + border * 2, image.height + border * 2), bg_color)
    canvas.paste(image, (border, border))
    if random.random() < 0.5:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=1.2))
        return canvas, f"screenshot_border{border}_blur"
    return canvas, f"screenshot_border{border}"


def generate_sns_augmentations(chain_depth: Sequence[int] | None = None, config: AugConfig | None = None) -> Callable[[Image.Image], Image.Image]:
    """SNS 스타일 증강 함수를 생성한다.

    Args:
        chain_depth: 각 시도에서 몇 개의 변형을 연쇄할지 범위를 지정.
        config: 증강 강도 설정.

    Returns:
        PIL 이미지를 입력받아 변형된 이미지를 반환하는 함수.
    """

    depth_range = chain_depth or (1, 3)
    if len(depth_range) != 2:
        raise ValueError("chain_depth는 [최소, 최대] 형태여야 합니다.")
    cfg = config or AugConfig()

    ops = [
        _resize_variant,
        _reencode,
        _resample_chain,
        _apply_noise_and_blur,
        _apply_tone,
        _add_watermark,
        _screenshot_simulation,
    ]

    def augment(image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image_rgb = image.convert("RGB")
        else:
            image_rgb = image.copy()

        history: List[str] = []
        num_steps = random.randint(depth_range[0], depth_range[1])
        selected_ops = random.sample(ops, k=min(num_steps, len(ops)))
        for op in selected_ops:
            # 재인코딩은 반복 적용될 수 있도록 확률적으로 다시 추가
            repeats = random.randint(*cfg.repeats)
            for _ in range(repeats):
                image_rgb, tag = op(image_rgb, cfg) if op in {_resize_variant, _reencode, _apply_noise_and_blur, _apply_tone, _add_watermark} else op(image_rgb)
                history.append(tag)
                if op not in {_resize_variant, _reencode, _apply_noise_and_blur, _apply_tone, _add_watermark}:
                    break
        augment.last_ops = history  # type: ignore[attr-defined]
        return image_rgb

    augment.last_ops = []  # type: ignore[attr-defined]
    return augment


def debug_augment(input_path: str, output_path: str) -> None:
    """단일 이미지를 증강하여 결과를 저장하고 로그로 체인을 확인한다."""

    image = Image.open(input_path).convert("RGB")
    augmenter = generate_sns_augmentations()
    augmented = augmenter(image)
    augmented.save(output_path)
    ops = getattr(augmenter, "last_ops", [])
    logger.info("SNS 증강 체인: %s", ", ".join(ops))
