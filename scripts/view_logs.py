from __future__ import annotations

import argparse
from pathlib import Path

from core.utils.logger import get_logger

logger = get_logger(__name__)


def _find_latest_experiment(model_name: str) -> Path:
    root = Path("experiments") / model_name
    if not root.exists():
        raise FileNotFoundError(f"experiment root not found: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no experiments recorded for model '{model_name}'")
    return sorted(candidates)[-1]


def _resolve_log_path(experiment_dir: Path, *, system: bool) -> Path:
    candidates = [experiment_dir / "logs" / "experiment.log"]
    legacy_name = "system.log" if system else "train.log"
    candidates.append(experiment_dir / "logs" / legacy_name)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"log file not found (checked: {', '.join(str(p) for p in candidates)})")


def view_train_logs(model_name: str, *, system: bool = False, tail: int | None = None) -> str:
    latest = _find_latest_experiment(model_name)
    log_path = _resolve_log_path(latest, system=system)
    with open(log_path, "r", encoding="utf-8") as fp:
        if tail is None or tail <= 0:
            return fp.read()
        lines = fp.readlines()
        return "".join(lines[-tail:])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View the latest training or system logs for a model.")
    parser.add_argument("model", help="model name (experiment folder under experiments/<model>/)")
    parser.add_argument("--system", action="store_true", help="show system.log instead of train.log")
    parser.add_argument("--tail", type=int, default=None, help="show only the last N lines")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        content = view_train_logs(args.model, system=args.system, tail=args.tail)
        print(content, end="")
    except FileNotFoundError as exc:
        logger.error(str(exc))


if __name__ == "__main__":
    main()
