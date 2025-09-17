# ---- 얼굴 검출 모델 파일 위치 ----
DEFAULT_DETECTOR_ONNX = \
    "/Users/sngmin/.cache/huggingface/hub/models--ykk648--face_lib/blobs/a3562ef62592bf387f6ef19151282ac127518e51c77696e62e0661bee95ba1ad"

VIDEO_PATH = '/Users/sngmin/gravifox/tvb-ai/sample.mp4'
DET_ONNX_PATH = DEFAULT_DETECTOR_ONNX
CLS_ONNX_PATH = '/Users/sngmin/.cache/huggingface/hub/models--prithivMLmods--Deepfake-Detection-Exp-02-22-ONNX/blobs/5b871f08a20f4543be3cec99eac74165821b6dc8f1b447c92391868e5d4f37b6'
CONF = 0.35
FPS = 30
CLIP_LEN = 1
CLIP_STRIDE = 1
ALIGN = 224
LAYOUT = 'NCTHW'
RGB= True
MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]
THRESHOLD=0.6
HIGH_CONF=0.8
SPECTRAL_R0=0.25
POSE_DELTA_OUTLIER=10
