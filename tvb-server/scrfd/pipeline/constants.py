from typing import Final
import numpy as np

# ---- 검출한 얼굴을 정렬할 때 기준이 되는 다섯 점(왼눈, 오른눈, 코, 왼입, 오른입)의 표준 위치 ----
# ---- 현재 프레임의 5점(랜드마크)와 이 기준 점을 맞춰서 얼굴을 정면에 가깝게 펴 줌 ----
REFERENCE_FIVE_POINTS: Final = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth
    [70.7299, 92.2041],  # right mouth
], dtype=np.float32)
REFERENCE_FIVE_POINTS.setflags(write=False)

# ---- 5개의 얼굴 랜드마크(눈/코/입)의 대략적인 3D 위치를 정의한 값 ----
# ---- solvePnP로 3D-2D 대응을 맞춰 얼굴의 회전(자세) 추정 ----
MODEL_3D_5PTS: Final = np.array([
    [-30.0,   0.0,  30.0],  # left eye
    [ 30.0,   0.0,  30.0],  # right eye
    [  0.0,   0.0,   0.0],  # nose tip
    [-25.0, -20.0,  20.0],  # left mouth
    [ 25.0, -20.0,  20.0],  # right mouth
], dtype=np.float32)
MODEL_3D_5PTS.setflags(write=False)
