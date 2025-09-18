import argparse
import importlib
from pathlib import Path
from typing import Tuple

import torch


def _load_sd(path: str):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state", "model_state_dict", "net", "module"):
            if k in obj:
                return obj[k]
    return obj  # may already be a state_dict


def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Export MINTIME extractor+classifier to TorchScript .pt files")
    ap.add_argument("--extractor-ckpt", required=True)
    ap.add_argument("--classifier-ckpt", required=True)
    ap.add_argument("--out-extractor", required=True)
    ap.add_argument("--out-classifier", required=True)
    ap.add_argument("--builder", required=True, help="pkg.module:build_fn returning (extractor, classifier)")
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    mod_name, fn_name = args.builder.split(":", 1)
    mod = importlib.import_module(mod_name)
    build = getattr(mod, fn_name)
    extractor, classifier = build()

    ext_sd = _load_sd(args.extractor_ckpt)
    cls_sd = _load_sd(args.classifier_ckpt)
    if isinstance(ext_sd, dict):
        try:
            extractor.load_state_dict(ext_sd, strict=False)
        except Exception:
            extractor.load_state_dict(ext_sd)
    if isinstance(cls_sd, dict):
        try:
            classifier.load_state_dict(cls_sd, strict=False)
        except Exception:
            classifier.load_state_dict(cls_sd)

    extractor.eval(); classifier.eval()

    # Dummy input (N,T,C,H,W) for tracing (prefer script; fallback trace)
    x = torch.randn(1, args.clip_len, 3, args.size, args.size, dtype=torch.float32)

    try:
        ext_ts = torch.jit.script(extractor)
    except Exception:
        ext_ts = torch.jit.trace(extractor, x, strict=False)

    with torch.no_grad():
        feats = extractor(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
    try:
        cls_ts = torch.jit.script(classifier)
    except Exception:
        cls_ts = torch.jit.trace(classifier, feats, strict=False)

    out_ext = Path(args.out_extractor)
    out_cls = Path(args.out_classifier)
    _ensure_parent(out_ext)
    _ensure_parent(out_cls)
    ext_ts.save(str(out_ext))
    cls_ts.save(str(out_cls))
    print(f"[EXPORT] saved extractor → {out_ext}")
    print(f"[EXPORT] saved classifier → {out_cls}")


if __name__ == "__main__":
    main()

