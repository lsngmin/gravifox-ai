"""E01–E16 실험 매트릭스 요약 스크립트(비파괴, 읽기 전용).

무엇을/왜:
    - 런처가 생성한 설정(`core/configs/experiments/vit_residual_matrix/vit_residual_E##.yaml`)
      목록과 실행된 실험 디렉터리(`experiments/residual_*`)를 "생성 순서=실행 순서"로 매핑.
    - 각 run의 `logs/train.log`를 파싱해 best val_acc/max, best val_loss/min, 마지막 에폭 지표를 집계.
    - 결과를 표로 출력하고 CSV로 저장(옵션).

제한사항:
    - 현재 run_name에 E##가 포함되지 않으므로, 순차 실행(런처의 작성 순서=실행 순서)을 가정합니다.
    - 도중 실패/누락 run이 있다면 매핑이 어긋날 수 있습니다.

사용 예시:
    python tvb-ai/scripts/summarize_vit_residual_matrix.py \
        --save-csv tvb-ai/experiments/matrix_summary.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import yaml

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViT-Residual 매트릭스 요약")
    parser.add_argument(
        "--config-dir",
        default=str(ROOT / "core/configs/experiments/vit_residual_matrix"),
        help="E## YAML들이 있는 디렉터리",
    )
    parser.add_argument(
        "--experiments-dir",
        default=str(ROOT / "experiments"),
        help="실험 아티팩트 루트",
    )
    parser.add_argument(
        "--model-name",
        default="residual",
        help="실험 디렉터리 접두어(예: residual_YYYY...)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="최근 N개 run만 요약(기본: 모두)",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="CSV 저장 경로(선택)",
    )
    return parser.parse_args()


def list_matrix_configs(config_dir: Path) -> List[Path]:
    configs = sorted(config_dir.glob("vit_residual_E*.yaml"))
    # Sort by E number
    def key(p: Path) -> int:
        m = re.search(r"E(\d{2})", p.name)
        return int(m.group(1)) if m else 9999

    return sorted(configs, key=key)


def read_hp_from_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train = cfg.get("train", {})
    opt = train.get("optimizer", {})
    meta = cfg.get("metadata", {})
    return {
        "experiment_id": meta.get("experiment_id"),
        "batch_size": train.get("batch_size"),
        "optimizer": opt.get("name"),
        "lr": opt.get("lr"),
        "weight_decay": opt.get("weight_decay"),
        "epochs": train.get("epochs"),
    }


def list_runs(experiments_dir: Path, model_name: str) -> List[Path]:
    runs = [p for p in experiments_dir.glob(f"{model_name}_*") if p.is_dir()]
    # Sort by timestamp suffix lexicographically (UTC format ensures chronological order)
    return sorted(runs, key=lambda p: p.name)


def parse_train_log(log_path: Path) -> Optional[Dict]:
    if not log_path.exists():
        return None
    best_acc = -1.0
    best_loss = float("inf")
    best_acc_epoch: Optional[int] = None
    best_loss_epoch: Optional[int] = None
    last_epoch: Optional[int] = None
    last_val_acc: Optional[float] = None
    last_val_loss: Optional[float] = None
    per_proc_bs: Optional[int] = None
    global_bs: Optional[int] = None

    # Patterns
    re_epoch = re.compile(
        r"에폭\s+(\d+)\s+완료\s+-\s+train_loss=([0-9.]+)\s+train_acc=([0-9.]+)\s+val_loss=([0-9.]+)\s+val_acc=([0-9.]+)"
    )
    re_accel = re.compile(r"Accelerate 초기화 .* per_proc_bs=(\d+), global_bs=(\d+)")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re_accel.search(line)
            if m:
                try:
                    per_proc_bs = int(m.group(1))
                    global_bs = int(m.group(2))
                except Exception:
                    pass
                continue

            m = re_epoch.search(line)
            if not m:
                continue
            ep = int(m.group(1))
            v_loss = float(m.group(4))
            v_acc = float(m.group(5))
            last_epoch = ep
            last_val_loss = v_loss
            last_val_acc = v_acc

            if v_acc > best_acc:
                best_acc = v_acc
                best_acc_epoch = ep
            if v_loss < best_loss:
                best_loss = v_loss
                best_loss_epoch = ep

    if last_epoch is None:
        return None
    return {
        "best_val_acc": best_acc,
        "best_val_acc_epoch": best_acc_epoch,
        "best_val_loss": best_loss,
        "best_val_loss_epoch": best_loss_epoch,
        "last_epoch": last_epoch,
        "last_val_acc": last_val_acc,
        "last_val_loss": last_val_loss,
        "per_proc_bs": per_proc_bs,
        "global_bs": global_bs,
    }


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir)
    exp_dir = Path(args.experiments_dir)

    cfg_paths = list_matrix_configs(config_dir)
    if not cfg_paths:
        print(f"[오류] 설정 디렉터리에 E## YAML이 없습니다: {config_dir}")
        sys.exit(1)

    # 읽은 순서가 E01..E16
    hp_list = [read_hp_from_yaml(p) | {"config_path": str(p)} for p in cfg_paths]

    run_paths = list_runs(exp_dir, args.model_name)
    if args.limit is not None:
        run_paths = run_paths[-args.limit :]

    n = min(len(hp_list), len(run_paths))
    if n == 0:
        print("[오류] 실행된 run 디렉터리를 찾지 못했습니다.")
        sys.exit(2)
    if len(hp_list) != len(run_paths):
        print(
            f"[경고] 설정({len(hp_list)})개와 run 디렉터리({len(run_paths)})개 수가 다릅니다. 앞에서부터 {n}개만 매핑합니다."
        )

    rows: List[Dict] = []
    for i in range(n):
        hp = hp_list[i]
        run = run_paths[i]
        log_path = run / "logs" / "train.log"
        parsed = parse_train_log(log_path)
        row: Dict = {
            "experiment_id": hp.get("experiment_id") or f"E{(i+1):02d}",
            "run_dir": str(run),
            "optimizer": hp.get("optimizer"),
            "weight_decay": hp.get("weight_decay"),
            "batch_size": hp.get("batch_size"),
            "lr": hp.get("lr"),
            "epochs": hp.get("epochs"),
        }
        if parsed is None:
            row.update({"status": "no_log"})
        else:
            row.update(parsed)
            row.update({"status": "ok"})
        rows.append(row)

    # Pretty print
    headers = [
        "experiment_id",
        "optimizer",
        "weight_decay",
        "batch_size",
        "lr",
        "best_val_acc",
        "best_val_loss",
        "best_val_acc_epoch",
        "best_val_loss_epoch",
        "last_epoch",
        "last_val_acc",
        "last_val_loss",
        "per_proc_bs",
        "global_bs",
        "run_dir",
        "status",
    ]
    print("\t".join(headers))
    for r in rows:
        def fmt(k: str) -> str:
            v = r.get(k)
            if isinstance(v, float):
                return f"{v:.4f}"
            return "" if v is None else str(v)

        print("\t".join(fmt(h) for h in headers))

    if args.save_csv:
        import csv

        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in rows:
                writer.writerow({h: r.get(h) for h in headers})
        print(f"CSV 저장 완료: {args.save_csv}")


if __name__ == "__main__":
    main()

