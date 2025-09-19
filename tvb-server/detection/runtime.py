import os, cv2, numpy as np
import onnxruntime as ort

def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px); preds.append(py)
    return np.stack(preds, axis=-1)

class SCRFDDetector:
    def __init__(self, onnx_path: str = None, session: ort.InferenceSession = None):
        assert onnx_path or session, "Provide onnx_path or session"
        self.session = session or ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self._init_vars()
        self.nms_thresh = 0.4
        self.input_size = (640, 640)  # (W,H) 고정 사용

    def _init_vars(self):
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        outs = self.session.get_outputs()
        self.batched = len(outs[0].shape) == 3  # 일부 onnx는 batch dim 포함
        self.output_names = [o.name for o in outs]
        self.use_kps = False
        self._num_anchors = 1

        # 출력 갯수로 stride/헤드 구성 추정 (insightface 구현 방식)
        if len(outs) == 6:
            self.fmc = 3; self._feat_stride_fpn = [8, 16, 32]; self._num_anchors = 2
        elif len(outs) == 9:
            self.fmc = 3; self._feat_stride_fpn = [8, 16, 32]; self._num_anchors = 2; self.use_kps = True
        elif len(outs) == 10:
            self.fmc = 5; self._feat_stride_fpn = [8, 16, 32, 64, 128]; self._num_anchors = 1
        elif len(outs) == 15:
            self.fmc = 5; self._feat_stride_fpn = [8, 16, 32, 64, 128]; self._num_anchors = 1; self.use_kps = True

        self.center_cache = {}

    def _blob(self, img):
        # SCRFD 표준 전처리: 1/128 스케일 + mean 127.5 + RGB
        blob = cv2.dnn.blobFromImage(
            img, 1.0/128, self.input_size, (127.5, 127.5, 127.5), swapRB=True
        )
        return blob

    def _forward_single(self, img, thresh=0.5):
        """img: 이미 (self.input_size)에 letterbox된 이미지"""
        blob = self._blob(img)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        H, W = blob.shape[2], blob.shape[3]

        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + self.fmc][0] * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + self.fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + self.fmc] * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + self.fmc * 2] * stride

            height, width = H // stride, W // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                grid = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (grid * stride).reshape(-1, 2)
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape(-1, 2)
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            # scores shape: (K * num_anchors,)
            pos_inds = np.where(scores.ravel() >= thresh)[0]
            if pos_inds.size == 0:
                continue

            bboxes = distance2bbox(anchor_centers, bbox_preds.reshape(-1, 4))
            pos_bboxes = bboxes[pos_inds]
            pos_scores = scores.ravel()[pos_inds]
            scores_list.append(pos_scores); bboxes_list.append(pos_bboxes)

            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds.reshape(-1, kps_preds.shape[1]))
                kpss = kpss.reshape(kpss.shape[0], -1, 2)
                kpss_list.append(kpss[pos_inds])

        return scores_list, bboxes_list, kpss_list

    @staticmethod
    def _nms(dets, thresh):
        x1, y1, x2, y2, scores = dets[:,0], dets[:,1], dets[:,2], dets[:,3], dets[:,4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def detect(self, img, conf_th=0.5, max_num=0):
        # letterbox to input_size (W,H)
        h, w = img.shape[:2]
        im_ratio = h / w
        model_ratio = self.input_size[1] / self.input_size[0]
        if im_ratio > model_ratio:
            new_h = self.input_size[1]; new_w = int(new_h / im_ratio)
        else:
            new_w = self.input_size[0]; new_h = int(new_w * im_ratio)
        det_scale = float(new_h) / h

        det_img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        resized = cv2.resize(img, (new_w, new_h))
        det_img[:new_h, :new_w, :] = resized

        scores_list, bboxes_list, kpss_list = self._forward_single(det_img, conf_th)
        if len(scores_list) == 0:
            return np.zeros((0,5), dtype=np.float32), None

        scores = np.hstack(scores_list)
        bboxes = np.vstack(bboxes_list) / det_scale
        order = scores.argsort()[::-1]
        pre_det = np.hstack([bboxes, scores.reshape(-1,1)]).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        keep = self._nms(pre_det, self.nms_thresh)
        det = pre_det[keep, :]

        kpss = None
        if self.use_kps and len(kpss_list) > 0:
            kpss = np.vstack(kpss_list) / det_scale
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]

        # 상위 max_num 제한 (선택)
        if max_num > 0 and det.shape[0] > max_num:
            areas = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
            center = np.array([h//2, w//2])
            offs = np.vstack([ (det[:,0]+det[:,2])/2 - center[1],
                               (det[:,1]+det[:,3])/2 - center[0] ])
            dist2 = np.sum(offs**2, axis=0)
            vals = areas - 2.0*dist2
            idx = np.argsort(vals)[::-1][:max_num]
            det = det[idx];
            if kpss is not None: kpss = kpss[idx]
        return det, kpss
