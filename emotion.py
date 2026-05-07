"""C 同学接口：情绪识别。"""

from __future__ import annotations

import base64
import logging
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    from deepface import DeepFace
except Exception:  # pragma: no cover
    DeepFace = None

EMOTION_DETECTOR_BACKENDS = ("retinaface", "mtcnn", "opencv")
DEFAULT_EMOTION = os.environ.get("EMOTION_DEFAULT", "neutral")
# 低置信度或 top1-top2 过小则降级为 neutral，避免随机抖动
EMOTION_MIN_CONFIDENCE = float(os.environ.get("EMOTION_MIN_CONFIDENCE", "0.45"))
EMOTION_MIN_MARGIN = float(os.environ.get("EMOTION_MIN_MARGIN", "0.10"))
EMOTION_ENABLE_AUG_VOTE = os.environ.get("EMOTION_ENABLE_AUG_VOTE", "1") != "0"
EMOTION_ANGRY_SURPRISE_MARGIN = float(os.environ.get("EMOTION_ANGRY_SURPRISE_MARGIN", "0.05"))


def _preprocess_lighting(image_bgr: np.ndarray) -> np.ndarray:
    """CLAHE 亮度均衡，缓解背光/暗光对情绪分类的影响。"""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    merged = cv2.merge([l_chan, a_chan, b_chan])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _build_emotion_views(image_bgr: np.ndarray) -> List[np.ndarray]:
    """
    构造多视图用于投票，降低单帧/单光照偶然误判。
    不做强旋转，避免破坏人脸几何结构。
    """
    base = image_bgr
    views: List[np.ndarray] = [base]
    # 轻度平滑，减少噪声对 micro-texture 的干扰
    views.append(cv2.GaussianBlur(base, (3, 3), 0))
    # 轻度提亮，修正暗光场景
    views.append(cv2.convertScaleAbs(base, alpha=1.06, beta=8))
    # 中心裁切后再缩放，降低背景干扰
    h, w = base.shape[:2]
    y1, y2 = int(h * 0.08), int(h * 0.92)
    x1, x2 = int(w * 0.08), int(w * 0.92)
    if y2 > y1 and x2 > x1:
        cropped = base[y1:y2, x1:x2]
        views.append(cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR))
    return views


def _decode_base64_image(image_base64: str) -> np.ndarray:
    """将 base64 图像字符串解码为 OpenCV 图像。"""
    if not image_base64:
        raise ValueError("图片数据为空")
    payload = image_base64.split(",")[-1].strip()
    if not payload:
        raise ValueError("图片数据为空")

    img_bytes = base64.b64decode(payload)
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("图片解码失败")
    return image


def _safe_area(face_row: Dict) -> int:
    region = face_row.get("region") or {}
    try:
        return int(region.get("w", 0)) * int(region.get("h", 0))
    except Exception:
        return 0


def _choose_primary_face(rows: List[Dict]) -> Optional[Dict]:
    if not rows:
        return None
    # 多人场景优先取面积最大的人脸，减少误取背景人脸
    return max(rows, key=_safe_area)


def _normalize_emotion_name(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if s in {"happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"}:
        return s
    return DEFAULT_EMOTION


def _pick_emotion_with_guard(emotion_scores: Dict) -> tuple[str, float]:
    normalized = {
        _normalize_emotion_name(k): float(v) / 100.0
        for k, v in (emotion_scores or {}).items()
    }
    if not normalized:
        return DEFAULT_EMOTION, 0.0

    ranked = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
    top_emotion, top_conf = ranked[0]
    second_conf = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_conf - second_conf
    if top_conf < EMOTION_MIN_CONFIDENCE or margin < EMOTION_MIN_MARGIN:
        return DEFAULT_EMOTION, top_conf
    return top_emotion, top_conf


def _aggregate_scores(score_maps: List[Dict[str, float]]) -> Dict[str, float]:
    if not score_maps:
        return {}
    merged: Dict[str, List[float]] = {}
    for m in score_maps:
        for k, v in m.items():
            merged.setdefault(k, []).append(float(v))
    # 使用均值 + 少量最大值，兼顾稳定性与明显表情峰值
    return {
        k: float(0.8 * np.mean(vals) + 0.2 * np.max(vals))
        for k, vals in merged.items()
        if vals
    }


def _calibrate_confusions(scores: Dict[str, float]) -> Dict[str, float]:
    """
    对常见混淆做轻量校正：
    - angry vs surprise: 惊讶常被误打成愤怒，若两者接近则给 surprise 小幅补偿
    """
    if not scores:
        return scores
    out = dict(scores)
    angry = out.get("angry", 0.0)
    surprise = out.get("surprise", 0.0)
    if angry > 0 and surprise > 0 and (angry - surprise) < EMOTION_ANGRY_SURPRISE_MARGIN:
        out["surprise"] = min(1.0, surprise + 0.04)
        out["angry"] = max(0.0, angry - 0.02)
    return out


def _extract_primary_emotion_scores(image: np.ndarray, backend: str) -> Optional[Dict[str, float]]:
    result = DeepFace.analyze(
        img_path=image,
        actions=["emotion"],
        detector_backend=backend,
        enforce_detection=False,
        align=True,
    )
    rows = result if isinstance(result, list) else [result]
    primary = _choose_primary_face(rows)
    if not primary:
        return None
    normalized = {
        _normalize_emotion_name(k): float(v) / 100.0
        for k, v in (primary.get("emotion", {}) or {}).items()
    }
    return normalized or None


def analyze_emotion(image_base64: str) -> dict:
    """分析人脸情绪，异常时降级为 neutral。"""
    try:
        image = _preprocess_lighting(_decode_base64_image(image_base64))
        if DeepFace is None:
            return {"emotion": DEFAULT_EMOTION, "confidence": 0.0}

        last_error = None
        views = _build_emotion_views(image) if EMOTION_ENABLE_AUG_VOTE else [image]
        for backend in EMOTION_DETECTOR_BACKENDS:
            try:
                score_maps: List[Dict[str, float]] = []
                for view in views:
                    scores = _extract_primary_emotion_scores(view, backend)
                    if scores:
                        score_maps.append(scores)
                if not score_maps:
                    continue

                agg_scores = _aggregate_scores(score_maps)
                calibrated = _calibrate_confusions(agg_scores)
                emotion, confidence = _pick_emotion_with_guard(
                    {k: v * 100.0 for k, v in calibrated.items()}
                )
                return {"emotion": emotion, "confidence": round(float(confidence), 4)}
            except Exception as exc:
                last_error = exc
                continue

        if last_error:
            LOGGER.warning("情绪分析回退到默认值: %s", last_error)
        return {"emotion": DEFAULT_EMOTION, "confidence": 0.0}
    except Exception as exc:
        LOGGER.exception("情绪分析失败: %s", exc)
        return {"emotion": DEFAULT_EMOTION, "confidence": 0.0}
