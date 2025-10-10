from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models.registry import get_model  # noqa: E402


def load_meta(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.yaml not found under {run_dir}")
    with meta_path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def resolve_state_dict(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")


def infer_image_size(meta: Dict[str, Any], fallback: int) -> int:
    dataset_cfg = meta.get("dataset") or {}
    for key in ("image_size", "img_size"):
        value = dataset_cfg.get(key)
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, (list, tuple)) and len(value) >= 1:
            try:
                return int(value[0])
            except Exception:
                continue
    loader_cfg = dataset_cfg.get("loader") or {}
    value = loader_cfg.get("image_size")
    if isinstance(value, int) and value > 0:
        return value
    return fallback


def build_model(meta: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_cfg = meta.get("model") or {}
    model_name = model_cfg.get("name")
    if not model_name:
        raise ValueError("meta.yaml does not contain model.name")
    params = model_cfg.get("params") or {}
    model = get_model(model_name, **params)
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained model to TorchScript using recorded training config.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to an experiment run directory (containing meta.yaml and checkpoints/).",
    )
    parser.add_argument(
        "--checkpoint",
        default="best.pt",
        help="Checkpoint filename under checkpoints/ to export (default: best.pt).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output TorchScript path (.pt). Defaults to <run-dir>/<model_name>_script.pt",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to load the model on during export (cpu | cuda).",
    )
    parser.add_argument(
        "--method",
        default="trace",
        choices=["trace", "script"],
        help="TorchScript export method (trace | script). Default: trace",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override input image size for dummy tensor. If omitted, meta.yaml dataset.image_size is used, falling back to 224.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run directory not found: {run_dir}")

    checkpoints_dir = run_dir / "checkpoints"
    ckpt_path = checkpoints_dir / args.checkpoint
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    meta = load_meta(run_dir)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(meta, device)
    state_dict = resolve_state_dict(ckpt_path, device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    image_size = args.image_size or infer_image_size(meta, fallback=224)
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    output_path = Path(
        args.output or run_dir / f"{meta.get('model', {}).get('name', 'model')}_torchscript.pt"
    ).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        if args.method == "script":
            scripted = torch.jit.script(model)
        else:
            scripted = torch.jit.trace(model, dummy, check_trace=False)
    scripted.save(str(output_path))
    print(f"TorchScript model saved to {output_path}")


if __name__ == "__main__":
    main()
