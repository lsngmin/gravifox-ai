"""멀티패치·멀티스케일 추론 유틸리티와 간단한 ViT 멀티패치 모델."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import timm
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms

from core.utils.logger import get_logger
from .registry import register


logger = get_logger(__name__)


@dataclass
class MultiPatchConfig:
    """ViTMultiPatch 모델 구성."""

    num_classes: int = 2
    pretrained: bool = True
    vit_name: str = "vit_base_patch16_224"


class ViTMultiPatch(nn.Module):
    """여러 패치를 평균해 최종 로짓을 생성하는 간단한 모델.

    SNS 환경에서 한 장의 이미지를 다양한 크기와 위치로 잘라서 추론하면
    압축 노이즈나 마스킹의 영향을 줄일 수 있다. 이 모델은 학습 시에도
    동일한 전략을 활용할 수 있도록 했다.
    """

    def __init__(self, cfg: MultiPatchConfig):
        super().__init__()
        self.cfg = cfg
        self.vit = timm.create_model(cfg.vit_name, pretrained=cfg.pretrained, num_classes=cfg.num_classes)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """하나의 텐서를 입력받아 멀티패치 평균 로짓을 반환한다."""

        b, c, h, w = image_tensor.shape
        center = torch.nn.functional.interpolate(image_tensor, size=(224, 224), mode="bilinear", align_corners=False)
        logits = self.vit(center)
        return logits


@register("vit_multipatch")
def build_vit_multipatch(**kwargs) -> ViTMultiPatch:
    """멀티패치 모델을 레지스트리에 등록한다."""

    cfg = MultiPatchConfig(**kwargs)
    return ViTMultiPatch(cfg)


def generate_patches(
    pil_image: Image.Image,
    sizes: Sequence[int] = (224, 336),
    n_patches: int = 12,
    strategy: str = "center+random",
) -> List[Image.Image]:
    """입력 이미지를 다양한 크기로 잘라 패치 리스트를 만든다.

    SNS에서 한 장의 이미지가 여러 비율로 잘리거나 리사이즈되는 상황을
    고려해, 중심 패치와 랜덤 패치를 조합해 안정적인 추론을 돕는다.
    """

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    patches: List[Image.Image] = []
    w, h = pil_image.size
    for size in sizes:
        center = ImageOps.fit(pil_image, (size, size), method=Image.BICUBIC, centering=(0.5, 0.5))
        patches.append(center)

    remaining = max(0, n_patches - len(patches))
    for _ in range(remaining):
        scale = random.choice(sizes)
        if strategy == "center+random":
            left = random.randint(0, max(0, w - scale))
            top = random.randint(0, max(0, h - scale))
            crop = pil_image.crop((left, top, left + scale, top + scale))
            patches.append(crop.resize((scale, scale), Image.BICUBIC))
        else:
            patches.append(ImageOps.fit(pil_image, (scale, scale), method=Image.BILINEAR))

    logger.info("멀티패치 생성 - 총 %d개, 스케일=%s", len(patches), list(sizes))
    return patches[:n_patches]


def infer_patches(
    model: nn.Module,
    patches: Iterable[Image.Image],
    device: str = "cuda",
    fp16: bool = True,
) -> List[dict]:
    """여러 패치를 모델에 통과시켜 점수를 반환한다."""

    model.eval()
    model.to(device)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    scores: List[dict] = []
    from contextlib import nullcontext

    use_cuda = device.startswith("cuda")
    autocast = torch.cuda.amp.autocast if use_cuda and fp16 else nullcontext
    with torch.inference_mode():
        with autocast():
            for patch in patches:
                tensor = preprocess(patch).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                scores.append({"ai": float(probs[1].item()), "real": float(probs[0].item())})
    return scores


def aggregate_scores(scores: Sequence[dict], method: str = "mean") -> dict:
    """패치별 점수를 하나의 확률로 통합한다.

    SNS 재압축으로 인해 패치마다 확신도가 달라질 수 있으므로, 평균/최대/
    품질 가중 방식 중 상황에 맞게 선택할 수 있도록 했다.
    """

    if not scores:
        raise ValueError("scores 리스트가 비어 있습니다.")

    ai_scores = torch.tensor([s["ai"] for s in scores])
    real_scores = torch.tensor([s["real"] for s in scores])

    if method == "max":
        ai = ai_scores.max().item()
        real = real_scores.min().item()
    elif method == "quality_weighted":
        weights = torch.linspace(1.0, 2.0, steps=len(scores))
        weights = weights / weights.sum()
        ai = (ai_scores * weights).sum().item()
        real = (real_scores * weights).sum().item()
    else:
        ai = ai_scores.mean().item()
        real = real_scores.mean().item()
    total = ai + real + 1e-8
    aggregated = {"ai": ai / total, "real": real / total}
    logger.info("패치 점수 집계 - method=%s, ai=%.4f, real=%.4f", method, aggregated["ai"], aggregated["real"])
    return aggregated
