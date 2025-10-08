"""학습 및 검증 지표 계산 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


@torch.no_grad()
def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 정확도를 계산한다."""

    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return float(correct) / float(max(total, 1))


def _confusion_binary(preds: np.ndarray, labels: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return tp, fp, tn, fn


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_ranks = ranks[labels == 1]
    auc = (pos_ranks.sum() - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def _expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(labels)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        accuracy = (predictions[mask] == labels[mask]).mean() if count else 0.0
        confidence = confidences[mask].mean() if count else 0.0
        ece += (count / total) * abs(accuracy - confidence)
    return float(ece)


def _tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr_target: float = 0.01) -> float:
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = scores.argsort()[::-1]
    labels_sorted = labels[order]
    tp = 0
    fp = 0
    best_tpr = 0.0
    for lbl in labels_sorted:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / max(neg, 1)
        tpr = tp / max(pos, 1)
        if fpr <= fpr_target:
            best_tpr = max(best_tpr, tpr)
        else:
            break
    return float(best_tpr)


@dataclass
class MetricBundle:
    loss: float
    acc: float
    f1: float
    auc: float
    ece: float
    tpr_at_1pct: float
    confusion: List[List[int]]

    def as_dict(self) -> Dict[str, float | List[List[int]]]:
        return {
            "loss": self.loss,
            "acc": self.acc,
            "f1": self.f1,
            "auc": self.auc,
            "ece": self.ece,
            "tpr@fpr=1%": self.tpr_at_1pct,
            "confusion": self.confusion,
        }


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> MetricBundle:
    """이진 분류용 확장 지표 집합."""

    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    labels = targets.detach().cpu().numpy().astype(int)
    preds = probs.argmax(axis=1)
    tp, fp, tn, fn = _confusion_binary(preds, labels)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-8)) if (tp + fp + fn) > 0 else 0.0

    metrics = MetricBundle(
        loss=0.0,
        acc=float((preds == labels).mean()),
        f1=float(f1),
        auc=_binary_auc(labels, probs[:, 1]),
        ece=_expected_calibration_error(probs, labels),
        tpr_at_1pct=_tpr_at_fpr(labels, probs[:, 1], fpr_target=0.01),
        confusion=[[int(tn), int(fp)], [int(fn), int(tp)]],
    )
    return metrics

