import numpy as np
import cv2

def preprocess_scrfd(frame, input_size=(640, 640)):
    h, w = frame.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))

    pad = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    pad[:nh, :nw] = resized

    blob = pad[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    blob = np.expand_dims(blob, 0) / 255.0
    return blob, scale, (w, h)

def run_scrfd(sess, frame):
    blob, scale, orig_shape = preprocess_scrfd(frame)
    input_name = sess.get_inputs()[0].name
    outs = sess.run(None, {input_name: blob})

    # ⚠️ SCRFD는 stride pyramid 기반이라 디코딩 로직이 필요
    # 여기서는 스켈레톤만 두고, 실제 decode_scrfd()는 레포 예제 참고 필요
    bboxes, landmarks, scores = decode_scrfd(outs, scale, orig_shape)
    return bboxes, landmarks, scores

def decode_scrfd(outs, scale, orig_shape):
    # TODO: anchors/stride별로 bbox 복원 + NMS
    raise NotImplementedError("SCRFD decode 구현 필요")
