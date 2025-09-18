import os
from typing import List, Optional, Tuple


class TorchClassifierRunner:
    """
    Minimal Torch runner that chains an Extractor and a Classifier head.
    - Prefers loading TorchScript via torch.jit.load for each stage.
    - If only state_dict checkpoints are provided, requires a builder hook
      via env TVB_TORCH_BUILDER="pkg.module:build_fn" that returns
      (extractor_module, classifier_module) when called.
    """

    def __init__(
        self,
        extractor_ckpt: str,
        model_ckpt: str,
        device: str = "cuda",
        rgb: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> None:
        try:
            import torch  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"PyTorch is required for Torch backend: {e}"
            )
        import torch

        self.torch = torch
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.rgb = rgb
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

        self.extractor, self.classifier = self._load_models(extractor_ckpt, model_ckpt)
        self.extractor.eval(); self.classifier.eval()
        self.extractor.to(self.device); self.classifier.to(self.device)

    # ----- Loading utils -----
    def _load_script(self, path: str):
        import torch
        try:
            return torch.jit.load(path, map_location=self.device)
        except Exception:
            return None

    def _load_state_dict(self, path: str):
        import torch
        try:
            obj = torch.load(path, map_location="cpu")
        except Exception:
            return None
        # common patterns: {'state_dict': ...} or raw state_dict
        if isinstance(obj, dict) and any(k in obj for k in ("state_dict", "model_state", "net", "module")):
            for k in ("state_dict", "model_state", "net", "module"):
                if k in obj:
                    return obj[k]
        if isinstance(obj, dict):
            return obj
        return None

    def _load_via_builder(self, ext_sd, cls_sd):
        hook = os.environ.get("TVB_TORCH_BUILDER")
        if not hook:
            raise RuntimeError(
                "Provide TVB_TORCH_BUILDER=package.module:build_fn to build models from state_dict."
            )
        mod_name, fn_name = hook.split(":", 1)
        mod = __import__(mod_name, fromlist=[fn_name])
        build = getattr(mod, fn_name)
        ext, cls = build()
        if ext_sd is not None:
            ext.load_state_dict(ext_sd, strict=False)
        if cls_sd is not None:
            cls.load_state_dict(cls_sd, strict=False)
        return ext, cls

    def _load_models(self, extractor_ckpt: str, model_ckpt: str):
        ext = self._load_script(extractor_ckpt)
        cls = self._load_script(model_ckpt)
        if ext is not None and cls is not None:
            return ext, cls

        ext_sd = self._load_state_dict(extractor_ckpt)
        cls_sd = self._load_state_dict(model_ckpt)
        if ext_sd is not None or cls_sd is not None:
            return self._load_via_builder(ext_sd, cls_sd)

        # last resort
        raise RuntimeError(
            "Failed to load Torch models. Ensure checkpoints are TorchScript .pt/.pth or provide a builder."
        )

    # ----- Preprocess -----
    def _to_tensor(self, frames: List, size: int, layout: str, rgb: bool,
                    mean: List[float], std: List[float]):
        import cv2
        import numpy as np
        T = len(frames)
        imgs = []
        for im in frames:
            x = im
            if rgb:
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = cv2.resize(x, (size, size))
            x = x.astype("float32") / 255.0
            imgs.append(x)
        arr = np.stack(imgs, axis=0)  # (T,H,W,3)
        arr = (arr - np.array(mean, dtype="float32")) / np.array(std, dtype="float32")
        arr = arr.transpose(0, 3, 1, 2)  # (T,3,H,W)
        # NCTHW expected
        if layout.upper() != "NCTHW":
            # convert to NCTHW
            # supported layouts in our pipeline are NCTHW/NCHW per-clip; fallback to NCTHW
            pass
        arr = arr[None, ...]  # (1,T,3,H,W)
        import torch
        return torch.from_numpy(arr).to(self.device)

    # ----- Inference -----
    def infer_clip(self, frames: List, size: int, layout: str, rgb: bool,
                   mean: Optional[List[float]] = None, std: Optional[List[float]] = None) -> Tuple[float, List[float]]:
        torch = self.torch
        x = self._to_tensor(frames, size=size, layout=layout, rgb=rgb,
                            mean=mean or self.mean, std=std or self.std)
        with torch.no_grad():
            feats = self.extractor(x)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            logits = self.classifier(feats)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().reshape(-1).tolist()
        return probs[0] if len(probs) else 0.0, probs

