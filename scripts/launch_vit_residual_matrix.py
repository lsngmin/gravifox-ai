"""ViT-Residual 실험 매트릭스 구성/실행 런처.

무엇을/왜:
    - 기본 YAML(`core/configs/vit_residual.yaml`)을 읽어 공통 설정과 데이터 경로를 유지한 채,
      하이퍼파라미터 조합(E01~E16)을 반영한 파생 설정 파일을 생성한다.
    - 옵션으로 각 실험을 순차 실행한다.

사용 예시:
    # 설정만 생성
    python tvb-ai/scripts/launch_vit_residual_matrix.py

    # 생성 후 순차 실행(싱글 GPU)
    python tvb-ai/scripts/launch_vit_residual_matrix.py --run

    # 멀티 GPU: accelerate로 2GPU 사용
    python tvb-ai/scripts/launch_vit_residual_matrix.py --run --launcher accelerate --num-proc 2 --gpus 0,1

    # 멀티 GPU: torchrun으로 4GPU 사용
    python tvb-ai/scripts/launch_vit_residual_matrix.py --run --launcher torchrun --num-proc 4 --gpus 0,1,2,3

생성 위치:
    core/configs/experiments/vit_residual_matrix/
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os

import yaml

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.utils.logger import get_logger


logger = get_logger(__name__)


def load_base_config(path: Path) -> Dict:
    """기준 YAML 설정을 로드한다.

    Args:
        path: 기준 YAML 경로

    Returns:
        설정 딕셔너리
    """

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: Dict, out_path: Path) -> None:
    """설정 딕셔너리를 YAML로 저장한다."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def apply_train_overrides(cfg: Dict, *, bs: int, lr: float, optim: str, wd: float) -> Dict:
    """학습 관련 오버라이드를 적용한다."""

    new_cfg = yaml.safe_load(yaml.safe_dump(cfg))  # deep copy via yaml
    train = new_cfg.setdefault("train", {})
    # Batch/LR/Optimizer
    train["batch_size"] = int(bs)
    opt = train.setdefault("optimizer", {})
    opt["name"] = str(optim)
    opt["lr"] = float(lr)
    opt["weight_decay"] = float(wd)
    # Scheduler: Cosine with warmup=5
    sch = train.setdefault("scheduler", {})
    sch["name"] = "cosine"
    sch["warmup_epochs"] = 5
    sch.setdefault("min_lr", 1.0e-6)
    # Regularization
    train["label_smoothing"] = float(new_cfg.get("train", {}).get("label_smoothing", 0.1))
    # Early stopping (monitor val_acc)
    early = train.setdefault("early_stopping", {})
    early["enabled"] = True
    early.setdefault("monitor", "val_acc")
    early.setdefault("patience", 10)
    early.setdefault("mode", "max")
    # Checkpoint monitor for best selection (val_acc)
    ck = train.setdefault("checkpoint", {})
    ck.setdefault("save_best", True)
    ck.setdefault("monitor", "val_acc")
    ck.setdefault("mode", "max")
    return new_cfg


def build_experiments() -> List[Tuple[str, Dict]]:
    """E01~E16 조합을 구성한다.

    Returns:
        (exp_id, overrides) 리스트. overrides는 {bs, lr, optim, wd} 키 포함
    """
    exps: List[Tuple[str, Dict]] = []

    # 512 배치, wd=0.01, AdamW/LAMB, lr in {5e-4, 1e-4, 5e-5}
    for i, lr in enumerate([5e-4, 1e-4, 5e-5], start=1):
        exps.append((f"E{str(i).zfill(2)}", {"bs": 512, "lr": lr, "optim": "adamw", "wd": 0.01}))
    for i, lr in enumerate([5e-4, 1e-4, 5e-5], start=4):
        exps.append((f"E{str(i).zfill(2)}", {"bs": 512, "lr": lr, "optim": "lamb", "wd": 0.01}))

    # 2048 배치, wd=0.01, AdamW/LAMB, lr in {2e-3, 4e-4, 2e-4}
    idx = 7
    for optim in ("adamw", "lamb"):
        for lr in (2e-3, 4e-4, 2e-4):
            exps.append((f"E{str(idx).zfill(2)}", {"bs": 2048, "lr": lr, "optim": optim, "wd": 0.01}))
            idx += 1

    # WD sensitivity checks
    exps.append(("E13", {"bs": 512, "lr": 1e-4, "optim": "adamw", "wd": 0.05}))
    exps.append(("E14", {"bs": 512, "lr": 1e-4, "optim": "lamb", "wd": 0.05}))
    exps.append(("E15", {"bs": 2048, "lr": 4e-4, "optim": "adamw", "wd": 0.05}))
    exps.append(("E16", {"bs": 2048, "lr": 4e-4, "optim": "lamb", "wd": 0.05}))

    return exps


def main() -> None:
    parser = argparse.ArgumentParser(description="ViT-Residual 실험 매트릭스 생성/실행")
    parser.add_argument(
        "--base-config",
        default=str(ROOT / "core/configs/vit_residual.yaml"),
        help="기준 YAML 경로",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "core/configs/experiments/vit_residual_matrix"),
        help="파생 YAML 저장 디렉터리",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="생성한 설정으로 순차 학습 실행",
    )
    parser.add_argument(
        "--launcher",
        choices=["python", "accelerate", "torchrun"],
        default="python",
        help="학습 실행 런처 선택 (멀티 GPU는 accelerate/torchrun 사용)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="가동 프로세스/디바이스 수(accelerate/torchrun)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="사용할 GPU 인덱스 목록 (예: '0,1'). 설정 시 CUDA_VISIBLE_DEVICES 적용",
    )
    parser.add_argument(
        "--main-process-ip",
        default="127.0.0.1",
        help="accelerate에서 메인 프로세스 IP를 강제 설정 (기본값: 127.0.0.1)",
    )
    parser.add_argument(
        "--main-process-port",
        type=int,
        default=29511,
        help="accelerate 메인 프로세스 포트 (기본값: 29511)",
    )
    parser.add_argument(
        "--disable-comm-tweaks",
        action="store_true",
        help="GLOO_SOCKET_IFNAME/NCCL_IB_DISABLE 기본 설정을 끕니다.",
    )
    args = parser.parse_args()

    base_path = Path(args.base_config)
    out_dir = Path(args.out_dir)

    cfg = load_base_config(base_path)
    exps = build_experiments()

    written: List[Path] = []
    for exp_id, ov in exps:
        new_cfg = apply_train_overrides(cfg, bs=ov["bs"], lr=ov["lr"], optim=ov["optim"], wd=ov["wd"])
        out_path = out_dir / f"vit_residual_{exp_id}.yaml"
        # 런 네임/로그에 식별자를 남기기 위한 주석성 태그
        new_cfg.setdefault("metadata", {})["experiment_id"] = exp_id
        save_config(new_cfg, out_path)
        written.append(out_path)
        logger.info("생성: %s -> %s", exp_id, out_path)

    if not args.run:
        logger.info("총 %d개 설정 생성 완료(실행 생략).", len(written))
        return

    # 순차 실행
    env = os.environ.copy()
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info("CUDA_VISIBLE_DEVICES=%s", args.gpus)

    for path in written:
        train_py = str(ROOT / "scripts" / "train.py")
        if args.launcher == "python":
            cmd = ["python", train_py, "--config", str(path)]
        elif args.launcher == "accelerate":
            cmd = [
                "accelerate",
                "launch",
                "--main_process_ip",
                str(args.main_process_ip),
                "--main_process_port",
                str(args.main_process_port),
                "--num_processes",
                str(max(1, args.num_proc)),
                train_py,
                "--config",
                str(path),
            ]
            if not args.disable_comm_tweaks:
                env.setdefault("GLOO_SOCKET_IFNAME", "lo")
                env.setdefault("NCCL_IB_DISABLE", "1")
                logger.info(
                    "Accelerate 통신 환경 변수 적용: GLOO_SOCKET_IFNAME=%s, NCCL_IB_DISABLE=%s",
                    env["GLOO_SOCKET_IFNAME"],
                    env["NCCL_IB_DISABLE"],
                )
        else:  # torchrun
            cmd = [
                "torchrun",
                "--nproc_per_node",
                str(max(1, args.num_proc)),
                train_py,
                "--config",
                str(path),
            ]

        logger.info("실행 시작: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
