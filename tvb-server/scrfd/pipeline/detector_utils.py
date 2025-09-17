
import cv2
import numpy as np
from typing import Tuple, Optional

from .constants import REFERENCE_FIVE_POINTS, MODEL_3D_5PTS

# ---- 랜드마크 5점으로 얼굴을 표준 위치로 정렬 ----
def warp_by_5pts(img: np.ndarray, kps: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    # 1) 입력 랜드마크를 float32, (5,2)로 보정
    src = np.asarray(kps, dtype=np.float32).reshape(5, 2)

    # 2) 출력 크기에 맞게 기준 5점 좌표(dst) 스케일 조정
    #    - ref_5pts는 112x112 기준이므로, 원하는 out_size에 맞춰 x,y 각각을 비율로 확대/축소
    dst = REFERENCE_FIVE_POINTS.copy()
    dst[:, 0] *= out_size[0] / 112.0
    dst[:, 1] *= out_size[1] / 112.0

    # 3) 부분 어핀 변환(affine) 행렬 추정
    #    - LMEDS는 외란(outlier)에 비교적 강인한 방법
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)

    # 4) 어핀 변환 적용(warpAffine)으로 정렬 이미지 생성
    aligned = cv2.warpAffine(img, M, out_size, borderValue=0)

    return aligned

# ---- 3D 기준점(MODEL_3D_5PTS)과 이미지 2D 랜드마크의 대응으로 solvePnP를 수행,
#         회전벡터(rvec)→회전행렬(R)로 변환해 오일러 각도를 계산한다. ----
def estimate_pose_5pts(kps: np.ndarray, img_shape: Tuple[int, int, int]) -> Optional[Tuple[float, float, float]]:
    """
    Return (yaw, pitch, roll) in degrees using 5pts.
    Use EPNP (>=4점 OK)로 초기 추정, 가능하면 ITERATIVE로 refine.
    """
    if kps is None or np.shape(kps) != (5, 2):
        return None

    h, w = img_shape[:2]

    # 간이 카메라 파라미터(초점/주점): 장면에 맞춰 조정 가능
    fx = fy = 0.8 * w
    cx, cy = w / 2.0, h / 2.0
    cam_mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    dist = np.zeros((4, 1), np.float32)  # 왜곡 0 가정

    # 3D-2D 대응
    obj3d = MODEL_3D_5PTS.astype(np.float32)
    img2d = kps.astype(np.float32).reshape(-1, 2)

    try:
        # 1) EPNP로 초기 추정 (최소 4점 필요)
        ok, rvec, tvec = cv2.solvePnP(
            obj3d, img2d, cam_mtx, dist,
            flags=cv2.SOLVEPNP_EPNP
        )
        if not ok:
            return None

        # 2) (옵션) ITERATIVE refine: 초기값을 넘겨 미세조정
        try:
            ok2, rvec, tvec = cv2.solvePnP(
                obj3d, img2d, cam_mtx, dist,
                rvec, tvec, True,  # useExtrinsicGuess=True
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            # ok2 실패해도 rvec/tvec는 EPNP 결과 유지
        except Exception:
            pass

        # 3) Rodrigues로 회전행렬, 오일러 각 계산
        R, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        yaw   = float(np.degrees(np.arctan2(R[2, 0], sy)))
        pitch = float(np.degrees(np.arctan2(-R[2, 1], R[2, 2])))
        roll  = float(np.degrees(np.arctan2(-R[1, 0], R[0, 0])))
        return yaw, pitch, roll
    except Exception:
        return None

# ---- 랜드마크(5점) 프레임 간 흔들림(평균 이동 거리)을 얼굴 크기로 정규화한 지표 ----
def lm_jitter(prev_kps: Optional[np.ndarray], curr_kps: Optional[np.ndarray], face_size: float) -> Optional[float]:
    if prev_kps is None or curr_kps is None:
        return None
    d = np.linalg.norm(prev_kps - curr_kps, axis=1).mean()
    return float(d / max(face_size, 1e-6))

"""
- 무엇: 2D FFT로 스펙트럼 크기(magnitude)를 계산한 뒤,
        중심(저주파) 기준 반경 r0 바깥(고주파)과 안쪽(저주파)의
        에너지 합을 구해 hi/(hi+lo) 비율을 반환한다.
- 입력:
  face_crop: BGR 얼굴 크롭(H,W,3)
  r0_ratio: 컷오프 반경 비율(0~0.5 근처 권장). 0.25면 반쪽 지름의 25% 지점.
- 출력:
  고주파 비율(0~1). 값이 클수록 고주파(자잘한 패턴/노이즈)가 상대적으로 많음을 의미.
- 주의:
  * 그레이스케일 변환 후 FFT를 적용.
  * lo(저주파 합)에 작은 epsilon을 더해 0 나눗셈 방지.
  * 전처리/해상도에 민감할 수 있으므로 r0_ratio는 데이터에 맞게 튜닝.
"""
def spectral_highfreq_ratio(face_crop: np.ndarray, r0_ratio: float = 0.25) -> float:
    # 1) 그레이스케일
    g = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    # 2) 2D FFT → 중심 정렬(fftshift)
    f = np.fft.fftshift(np.fft.fft2(g.astype(np.float32)))
    mag = np.abs(f)

    # 3) 원형 마스크를 위한 좌표/반경 계산
    H, W = mag.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.hypot(yy - cy, xx - cx)

    # 4) 컷오프 반경 r0 설정(최소 반쪽 지름 기준의 비율)
    r0 = r0_ratio * (min(H, W) / 2)

    # 5) 고주파/저주파 영역 에너지 합
    hi = mag[r >= r0].sum()
    lo = mag[r < r0].sum() + 1e-6

    # 6) 고주파 에너지 비율
    return float(hi / (hi + lo))