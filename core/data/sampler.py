"""여러 데이터셋을 혼합 샘플링하여 학습/검증 세트를 생성하는 모듈."""

from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from core.utils.logger import get_logger


logger = get_logger(__name__)
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _is_image_file(path: Path) -> bool:
    """지원 형식 여부를 검사한다.

    Args:
        path: 확인할 경로.

    Returns:
        이미지 파일이면 True, 아니면 False.
    """

    return path.is_file() and path.suffix.lower() in _ALLOWED_SUFFIXES


def _collect_class_files(dataset_path: Path) -> Dict[str, List[Path]]:
    """ImageFolder 구조에서 클래스별 파일 목록을 수집한다.

    Args:
        dataset_path: ImageFolder 형태의 루트 경로.

    Returns:
        클래스 이름을 키로 하는 파일 경로 리스트 딕셔너리.
    """

    class_to_files: Dict[str, List[Path]] = defaultdict(list)
    for class_dir in dataset_path.iterdir():
        if not class_dir.is_dir():
            continue
        for file_path in class_dir.rglob("*"):
            if _is_image_file(file_path):
                class_to_files[class_dir.name].append(file_path)
    return class_to_files


def _allocate_counts(total_target: int, class_map: Dict[str, List[Path]]) -> Dict[str, int]:
    """클래스별 샘플 개수를 비율에 따라 분배한다.

    Args:
        total_target: 전체로 확보하고 싶은 샘플 수.
        class_map: 클래스별 파일 리스트.

    Returns:
        클래스별 샘플 개수 딕셔너리.
    """

    counts: Dict[str, int] = {}
    total_available = sum(len(files) for files in class_map.values())
    if total_available == 0:
        return {cls: 0 for cls in class_map}

    remainder: List[Tuple[str, float]] = []
    assigned = 0
    for class_name, files in class_map.items():
        proportion = len(files) / total_available
        raw = proportion * total_target
        count = min(len(files), int(raw))
        counts[class_name] = count
        assigned += count
        remainder.append((class_name, raw - count))

    remainder.sort(key=lambda item: item[1], reverse=True)
    idx = 0
    while assigned < total_target and any(len(class_map[name]) > counts[name] for name, _ in remainder):
        name, _ = remainder[idx % len(remainder)]
        if counts[name] < len(class_map[name]):
            counts[name] += 1
            assigned += 1
        idx += 1
    return counts


def _sample_class_files(
    dataset_name: str,
    class_map: Dict[str, List[Path]],
    target_per_class: Dict[str, int],
    rng: random.Random,
) -> List[Tuple[str, Path, str]]:
    """클래스별 목표 개수만큼 무작위 샘플을 선택한다.

    Args:
        dataset_name: 데이터셋 식별자(파일명 충돌 방지용).
        class_map: 클래스별 파일 리스트.
        target_per_class: 클래스별 샘플 목표 수.
        rng: 난수 생성기.

    Returns:
        (클래스, 경로, 데이터셋명) 튜플 리스트.
    """

    selected: List[Tuple[str, Path, str]] = []
    for class_name, files in class_map.items():
        target = target_per_class.get(class_name, 0)
        if target <= 0:
            continue
        if target >= len(files):
            chosen = files
        else:
            chosen = rng.sample(files, target)
        for file_path in chosen:
            selected.append((class_name, file_path, dataset_name))
    rng.shuffle(selected)
    return selected


def _split_train_val(samples: List[Tuple[str, Path, str]], train_ratio: float) -> Tuple[List[Tuple[str, Path, str]], List[Tuple[str, Path, str]]]:
    """샘플 리스트를 학습/검증으로 분할한다.

    Args:
        samples: 선택된 샘플 목록.
        train_ratio: 학습 데이터 비율(0~1).

    Returns:
        (train_samples, val_samples) 튜플.
    """

    train_size = int(len(samples) * train_ratio)
    train_split = samples[:train_size]
    val_split = samples[train_size:]
    return train_split, val_split


def _copy_samples(target_dir: Path, split_name: str, samples: List[Tuple[str, Path, str]]) -> None:
    """선택된 샘플을 출력 디렉터리로 복사한다.

    Args:
        target_dir: 출력 루트 경로.
        split_name: train 또는 val.
        samples: (클래스, 경로, 데이터셋명) 리스트.
    """

    for index, (class_name, file_path, dataset_name) in enumerate(samples):
        dest_dir = target_dir / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_name = f"{dataset_name}_{index}_{file_path.name}"
        shutil.copy2(file_path, dest_dir / dest_name)


def sample_datasets(config: dict) -> None:
    """여러 데이터셋을 혼합 샘플링하여 학습/검증 세트를 생성한다.

    Args:
        config: YAML data 섹션(데이터셋, 출력 경로, 샘플링 설정 포함).

    Returns:
        None
    """

    datasets_cfg = config.get("datasets", [])
    if not datasets_cfg:
        logger.info("데이터셋 구성이 비어 있어 샘플링을 건너뜁니다.")
        return

    output_dir = Path(config.get("output_dir", "./mixed"))
    train_dir = Path(config.get("train_dir", output_dir / "train"))
    val_dir = Path(config.get("val_dir", output_dir / "val"))
    sampling_cfg = config.get("sampling", {})
    policy = sampling_cfg.get("policy", "once").lower()
    train_ratio = float(sampling_cfg.get("ratio", 0.8))
    train_ratio = min(max(train_ratio, 0.0), 1.0)
    sampling_seed = int(sampling_cfg.get("seed", 1337))

    if policy == "once" and train_dir.exists() and any(train_dir.rglob("*")):
        logger.info("샘플링 정책 once - 기존 혼합 데이터셋을 재사용합니다.")
        return

    if policy == "always" and output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info("샘플링 정책 always - 기존 혼합 데이터셋을 삭제했습니다.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(sampling_seed)

    dataset_infos = []
    total_weight = 0.0
    for entry in datasets_cfg:
        dataset_path = Path(entry["path"])
        if not dataset_path.exists():
            logger.error("데이터셋 경로를 찾을 수 없습니다: %s", dataset_path)  # TODO: 필요 없을 시 삭제 가능
            raise FileNotFoundError(f"Dataset path missing: {dataset_path}")
        class_map = _collect_class_files(dataset_path)
        total_files = sum(len(files) for files in class_map.values())
        if total_files == 0:
            logger.error("데이터셋이 비어 있습니다: %s", dataset_path)  # TODO: 필요 없을 시 삭제 가능
            raise ValueError(f"Dataset empty: {dataset_path}")
        weight = float(entry.get("weight", 1.0))
        if weight <= 0:
            continue
        dataset_infos.append({
            "name": entry.get("name", dataset_path.name),
            "class_map": class_map,
            "weight": weight,
            "total_files": total_files,
        })
        total_weight += weight

    if not dataset_infos:
        logger.error("샘플링 가능한 데이터셋이 없습니다.")  # TODO: 필요 없을 시 삭제 가능
        raise ValueError("No datasets to sample")

    candidate_totals = [info["total_files"] * total_weight / info["weight"] for info in dataset_infos]
    global_target = int(min(candidate_totals))
    if global_target <= 0:
        logger.error("샘플링 가능한 데이터가 없습니다.")  # TODO: 필요 없을 시 삭제 가능
        raise ValueError("global_target must be positive")

    logger.info("혼합 샘플링 시작 - target=%d, train_ratio=%.2f", global_target, train_ratio)

    for info in dataset_infos:
        desired_total = int(info["weight"] / total_weight * global_target)
        if desired_total <= 0:
            logger.warning("데이터셋 %s에서 샘플링할 수 있는 양이 부족합니다.", info["name"])
            continue
        target_per_class = _allocate_counts(desired_total, info["class_map"])
        samples = _sample_class_files(info["name"], info["class_map"], target_per_class, rng)
        train_split, val_split = _split_train_val(samples, train_ratio)
        _copy_samples(output_dir, "train", train_split)
        _copy_samples(output_dir, "val", val_split)
        logger.info(
            "데이터셋 %s 샘플링 - train=%d, val=%d", info["name"], len(train_split), len(val_split)
        )

    logger.info("혼합 샘플링 완료 - 출력 경로: %s", output_dir)
