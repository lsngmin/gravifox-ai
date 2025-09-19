from typing import Optional, List

import cv2
import numpy as np

# ---- 얼굴 크롭 이미지를 분류기 입력 텐서(NCHW)로 바꾸는 전처리 ----
def preprocess_image(img: np.ndarray, size: int = 224, rgb: bool = True,
                     mean: Optional[List[float]] = None, std: Optional[List[float]] = None) -> np.ndarray:
    """Aligned BGR face -> (1,3,H,W) float32 (NCHW)."""
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
    if mean is not None and std is not None:
        img = (img - np.array(mean, np.float32)) / np.array(std, np.float32)
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


"""
- 흐름: 리사이즈(size×size) → (선택)BGR→RGB → [0,1] 스케일 → (선택)정규화 → 레이아웃 변환
- 입력:
  frames: BGR 프레임 리스트 [H,W,3] (길이 T)
  size: 한 변 크기(정방형)
  layout: "NCTHW" 또는 "NTHWC" (모델이 기대하는 입력 형상)
  rgb: True면 BGR→RGB 변환
  mean/std: [R,G,B] 정규화(0~1 스케일 기준). 없으면 생략
- 출력:
  layout=="NCTHW": (1, 3, T, H, W) float32
  layout=="NTHWC": (1, T, H, W, 3) float32
- 주의:
  * NCTHW = (N, C, T, H, W), NTHWC = (N, T, H, W, C)
  * 모델 스펙(채널 순서/정규화)과 일치하도록 mean/std와 rgb를 맞출 것
"""
def preprocess_frames(frames: List[np.ndarray], size: int, layout: str, rgb: bool,
                      mean: Optional[List[float]], std: Optional[List[float]]) -> np.ndarray:
    """frames -> ONNX input tensor for video models.
    layout: 'NCTHW' or 'NTHWC'. mean/std in 0-1 scale.
    """
    proc = []
    for f in frames:
        img = cv2.resize(f, (size, size))
        if rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        if mean is not None and std is not None:
            img = (img - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
        proc.append(img)
    arr = np.stack(proc, axis=0)  # (T,H,W,3)
    if layout.upper() == "NCTHW":
        arr = np.transpose(arr, (0, 3, 1, 2))  # (T,3,H,W)
        arr = arr[None, ...]                    # (1,T,3,H,W)
        arr = np.transpose(arr, (0, 2, 1, 3, 4))# (1,3,T,H,W)
    elif layout.upper() == "NTHWC":
        arr = arr[None, ...]                    # (1,T,H,W,3)
    else:
        raise ValueError("Unsupported layout. Use NCTHW or NTHWC")
    return arr.astype(np.float32)