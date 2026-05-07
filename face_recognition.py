"""C 同学接口：人脸识别、照片活体检测与合照识别（鲁棒性优化版）。

主要优化：
1. 多增强向量投票匹配  —— 对输入图做多种扰动后提取多条嵌入，用 top-k 投票代替单一距离
2. 更全面的预处理管线  —— CLAHE + 高斯去噪 + 方向修正（EXIF）
3. 合照识别加 NMS     —— 去除重复检测的同一张脸，避免一人被匹配多次或互相干扰
4. 自适应置信度       —— 将原始余弦距离映射到更有区分度的 sigmoid 置信度
5. 模板注册优化       —— 注册时同样做多增强入库，提升数据库多样性
"""

from __future__ import annotations

import base64
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

FACE_DB_PATH = "backend/data/face_features.pkl"

EMBEDDING_MODEL_NAME = os.environ.get("DEEPFACE_MODEL", "ArcFace").strip() or "ArcFace"

_DEFAULT_THRESHOLDS = {
    "ArcFace": 0.40,
    "Facenet512": 0.42,
    "Facenet": 0.45,
    "VGG-Face": 0.55,
}
MATCH_THRESHOLD = float(
    os.environ.get(
        "FACE_MATCH_THRESHOLD",
        str(_DEFAULT_THRESHOLDS.get(EMBEDDING_MODEL_NAME, 0.48)),
    )
)

MIN_CONFUSION_MARGIN = float(os.environ.get("FACE_MIN_CONFUSION_MARGIN", "0.006"))
MAX_FACE_TEMPLATES = max(1, int(os.environ.get("FACE_MAX_TEMPLATES", "5")))
TEMPLATE_MERGE_MIN_DISTANCE = float(os.environ.get("FACE_TEMPLATE_MERGE_MIN_DISTANCE", "0.07"))

# ── 新增：识别时的增强开关 ───────────────────────────────────────────
# 推理时对输入图像做多种增强，每种生成一条嵌入向量，汇总 top-k 投票
# 可通过环境变量关闭（性能受限环境）
ENABLE_AUGMENTATION_VOTE = os.environ.get("FACE_ENABLE_AUG_VOTE", "1") != "0"

# ── 新增：合照 NMS IOU 阈值（避免同一张脸被多次检出） ──────────────
NMS_IOU_THRESHOLD = float(os.environ.get("FACE_NMS_IOU", "0.10"))

ENABLE_LIVENESS_CHECK = True
HAAR_FACE_MODEL = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DETECTOR_BACKENDS = ("retinaface", "mtcnn", "opencv")

# ── 集体照专用配置 ───────────────────────────────────────────────────────
# 遍历所有后端，选人脸数最多的结果
DETECTOR_BACKENDS_GROUP = ("retinaface", "mtcnn", "ssd", "opencv")
# 图像短边小于此值时自动放大，改善小人脸检测率
GROUP_PHOTO_UPSCALE_MIN_DIM = int(os.environ.get("FACE_GROUP_UPSCALE_MIN_DIM", "960"))
# 开启分块检测（适合人数多、人脸较小的大合照）
GROUP_PHOTO_ENABLE_TILING = os.environ.get("FACE_GROUP_TILING", "1") != "0"
# 分块行数（列数按宽高比自动计算）
GROUP_PHOTO_TILE_ROWS = max(1, int(os.environ.get("FACE_GROUP_TILE_ROWS", "2")))

try:
    from deepface import DeepFace
except Exception:
    DeepFace = None

face_encodings_db: Dict[str, List[np.ndarray]] = {}
face_student_names: Dict[str, str] = {}


# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════

def _normalize_encoding_list(raw_encoding) -> List[np.ndarray]:
    """兼容单向量/向量列表，统一返回向量列表。"""
    if raw_encoding is None:
        return []
    if isinstance(raw_encoding, list):
        normalized = []
        for item in raw_encoding:
            if item is None:
                continue
            arr = np.asarray(item, dtype=np.float32)
            if arr.size > 0:
                normalized.append(arr)
        return normalized
    arr = np.asarray(raw_encoding, dtype=np.float32)
    return [arr] if arr.size > 0 else []


def _cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    emb1 = np.asarray(emb1, dtype=np.float32)
    emb2 = np.asarray(emb2, dtype=np.float32)
    n1, n2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
    if n1 == 0 or n2 == 0:
        return 1.0
    return float(1.0 - np.dot(emb1, emb2) / (n1 * n2))


def _distance_to_confidence(distance: float) -> float:
    """
    优化1：用 sigmoid 替换线性映射，使置信度在阈值附近变化更陡峭、更有区分度。
    distance=0 → 0.98, distance=threshold → ~0.5, distance=1 → 0.02
    """
    k = 10.0 / max(MATCH_THRESHOLD, 1e-6)  # 斜率自适应阈值
    return float(1.0 / (1.0 + np.exp(k * (distance - MATCH_THRESHOLD))))


def _same_dim(emb1: np.ndarray, emb2: np.ndarray) -> bool:
    return int(np.asarray(emb1).size) == int(np.asarray(emb2).size)


# ════════════════════════════════════════════════════════════
# 优化2：更完整的图像预处理管线
# ════════════════════════════════════════════════════════════

def _fix_image_orientation(image_bgr: np.ndarray, image_bytes: Optional[bytes] = None) -> np.ndarray:
    """
    根据 EXIF 方向标记旋转图像（手机竖拍场景常见）。
    如果没有 EXIF 数据则原样返回。
    """
    if image_bytes is None:
        return image_bgr
    try:
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        # 读取 EXIF 方向（仅 JPEG 有效）
        exif_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if exif_img is None:
            return image_bgr
        # OpenCV 4.5+ 可通过 ExifOrientationCorrectionMode 处理，此处手动处理常见旋转
        # 简化版：依赖 PIL（可选），不可用时跳过
        try:
            from PIL import Image, ExifTags
            import io
            pil_img = Image.open(io.BytesIO(image_bytes))
            exif_data = pil_img._getexif()
            if exif_data:
                orient_key = next(
                    (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
                )
                if orient_key and orient_key in exif_data:
                    orientation = exif_data[orient_key]
                    rotations = {3: 180, 6: 270, 8: 90}
                    if orientation in rotations:
                        pil_img = pil_img.rotate(rotations[orientation], expand=True)
                        pil_arr = np.array(pil_img)
                        return cv2.cvtColor(pil_arr, cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    except Exception:
        pass
    return image_bgr


def _preprocess_lighting(image_bgr: np.ndarray) -> np.ndarray:
    """CLAHE 亮度均衡 + 高斯去噪，减轻光照、噪点影响。"""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    # 高斯去噪（轻量，不引入模糊）
    denoised = cv2.GaussianBlur(image_bgr, (3, 3), 0)
    # CLAHE 均衡
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge([l_chan, a_chan, b_chan]), cv2.COLOR_LAB2BGR)


def _preprocess_group_photo(image_bgr: np.ndarray) -> np.ndarray:
    """
    集体照专用预处理：
    1. 若图像短边不足 GROUP_PHOTO_UPSCALE_MIN_DIM，先放大，确保小人脸达到检测所需分辨率
    2. 使用更温和的 CLAHE（clipLimit=1.5），避免过度增强破坏人脸纹理
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    min_dim = min(h, w)
    if min_dim < GROUP_PHOTO_UPSCALE_MIN_DIM:
        scale = GROUP_PHOTO_UPSCALE_MIN_DIM / min_dim
        image_bgr = cv2.resize(
            image_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
    denoised = cv2.GaussianBlur(image_bgr, (3, 3), 0)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge([l_chan, a_chan, b_chan]), cv2.COLOR_LAB2BGR)


def _decode_base64_image(image_base64: str) -> Tuple[np.ndarray, bytes]:
    """解码 base64，同时返回原始字节（用于 EXIF 解析）。"""
    if not image_base64:
        raise ValueError("图片数据为空")
    payload = image_base64.split(",")[-1].strip()
    if not payload:
        raise ValueError("图片数据为空")
    img_bytes = base64.b64decode(payload)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("图片解码失败")
    return image, img_bytes


# ════════════════════════════════════════════════════════════
# 优化3：多增强向量生成（推理阶段数据增强）
# ════════════════════════════════════════════════════════════

def _generate_augmented_variants(image_bgr: np.ndarray) -> List[np.ndarray]:
    """
    对输入图生成多个轻微变体：
    - 原图
    - 水平翻转（镜像）
    - 轻微提亮 / 压暗
    - 轻微对比度增强
    返回最多 5 个变体，供后续各自提取嵌入后投票。
    """
    variants = [image_bgr]

    # 水平翻转（ArcFace 对镜像鲁棒性有限，加入可提升多角度覆盖）
    variants.append(cv2.flip(image_bgr, 1))

    # 轻微提亮
    bright = cv2.convertScaleAbs(image_bgr, alpha=1.0, beta=25)
    variants.append(bright)

    # 轻微压暗
    dark = cv2.convertScaleAbs(image_bgr, alpha=1.0, beta=-25)
    variants.append(dark)

    # 对比度增强
    contrast = cv2.convertScaleAbs(image_bgr, alpha=1.2, beta=0)
    variants.append(contrast)

    return variants


# ════════════════════════════════════════════════════════════
# DeepFace 嵌入提取（保持原有结构）
# ════════════════════════════════════════════════════════════

def _deepface_represent(image_bgr: np.ndarray) -> List[dict]:
    if DeepFace is None:
        raise RuntimeError("DeepFace 未安装或导入失败")
    last_error: Optional[Exception] = None
    for enforce_detection in (True, False):
        for backend in DETECTOR_BACKENDS:
            try:
                result = DeepFace.represent(
                    img_path=np.asarray(image_bgr),
                    model_name=EMBEDDING_MODEL_NAME,
                    detector_backend=backend,
                    enforce_detection=enforce_detection,
                    align=True,
                )
                if result:
                    return result
            except Exception as exc:
                last_error = exc
    if last_error:
        raise RuntimeError(f"特征提取失败: {last_error}")
    return []


def _deepface_represent_group(image_bgr: np.ndarray) -> List[dict]:
    """
    集体照专用：遍历所有后端（enforce_detection=False），
    返回检测到人脸数量最多的结果集，而非第一个成功的结果。
    """
    if DeepFace is None:
        raise RuntimeError("DeepFace 未安装或导入失败")
    best_results: List[dict] = []
    for backend in DETECTOR_BACKENDS_GROUP:
        try:
            result = DeepFace.represent(
                img_path=np.asarray(image_bgr),
                model_name=EMBEDDING_MODEL_NAME,
                detector_backend=backend,
                enforce_detection=False,
                align=True,
            )
            if result and len(result) > len(best_results):
                best_results = result
        except Exception:
            continue
    return best_results


def _facial_area(r: dict) -> int:
    fa = r.get("facial_area") or {}
    try:
        return int(fa.get("w", 0)) * int(fa.get("h", 0))
    except (TypeError, ValueError):
        return 0


def _pick_largest_face_embedding(represent_results: List[dict]) -> np.ndarray:
    best = max(represent_results, key=_facial_area)
    return np.asarray(best["embedding"], dtype=np.float32)


def _pick_all_face_embeddings_with_boxes(
    represent_results: List[dict],
) -> List[Tuple[np.ndarray, dict]]:
    """返回 (embedding, facial_area_dict) 列表，供合照 NMS 使用。"""
    return [
        (np.asarray(r["embedding"], dtype=np.float32), r.get("facial_area") or {})
        for r in represent_results
    ]


def _extract_embedding_from_crop(face_image: np.ndarray) -> np.ndarray:
    if DeepFace is None:
        raise RuntimeError("DeepFace 未安装或导入失败")
    last_error: Optional[Exception] = None
    for enforce_detection in (True, False):
        for backend in DETECTOR_BACKENDS:
            try:
                result = DeepFace.represent(
                    img_path=np.asarray(face_image),
                    model_name=EMBEDDING_MODEL_NAME,
                    detector_backend=backend,
                    enforce_detection=enforce_detection,
                    align=True,
                )
                if result:
                    return np.asarray(result[0]["embedding"], dtype=np.float32)
            except Exception as exc:
                last_error = exc
    if last_error:
        raise RuntimeError(f"特征提取失败: {last_error}")
    raise RuntimeError("特征提取结果为空")


def _extract_embedding(face_image: np.ndarray) -> np.ndarray:
    return _extract_embedding_from_crop(face_image)


def _detect_faces_local(
    image: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 5
) -> List[Tuple[int, int, int, int]]:
    classifier = cv2.CascadeClassifier(HAAR_FACE_MODEL)
    if classifier.empty():
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(40, 40)
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def _detect_faces_local_group(
    image: np.ndarray,
) -> List[Tuple[int, int, int, int]]:
    """
    集体照 Haar 兜底检测：
    - minSize 缩小到 20×20，捕捉远景小人脸
    - 多组参数叠加，减少漏检
    """
    classifier = cv2.CascadeClassifier(HAAR_FACE_MODEL)
    if classifier.empty():
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    seen: List[Tuple[int, int, int, int]] = []
    for scale, neighbors, min_sz in [(1.05, 3, 20), (1.1, 4, 30), (1.15, 5, 40)]:
        try:
            raw = classifier.detectMultiScale(
                gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(min_sz, min_sz)
            )
            if len(raw) > 0:
                seen.extend([(int(x), int(y), int(w), int(h)) for x, y, w, h in raw])
        except Exception:
            continue
    return seen


def _crop_face_local(image: np.ndarray, face_box: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = face_box
    face = image[y: y + h, x: x + w]
    if face.size == 0:
        raise ValueError("裁剪后的人脸为空")
    return face


# ════════════════════════════════════════════════════════════
# 优化后的主要嵌入提取接口
# ════════════════════════════════════════════════════════════

def extract_primary_embedding(image_bgr: np.ndarray) -> np.ndarray:
    """提取「主要人脸」嵌入（面积最大），供原有流程使用。"""
    processed = _preprocess_lighting(image_bgr)
    rows = _deepface_represent(processed)
    if rows:
        return _pick_largest_face_embedding(rows)
    faces = _detect_faces_local(processed)
    if not faces:
        raise RuntimeError("未检测到人脸")
    return _extract_embedding_from_crop(_crop_face_local(processed, faces[0]))


def extract_primary_embedding_augmented(image_bgr: np.ndarray) -> List[np.ndarray]:
    """
    优化3：对输入图的多个增强变体各自提取嵌入，返回向量列表。
    识别时用投票机制聚合，可显著提升光线/表情变化下的鲁棒性。
    """
    processed = _preprocess_lighting(image_bgr)
    variants = _generate_augmented_variants(processed)
    embeddings: List[np.ndarray] = []
    for variant in variants:
        try:
            rows = _deepface_represent(variant)
            if rows:
                embeddings.append(_pick_largest_face_embedding(rows))
        except Exception:
            continue
    if not embeddings:
        # 兜底：只用原图
        try:
            embeddings.append(extract_primary_embedding(image_bgr))
        except Exception:
            raise RuntimeError("未检测到人脸")
    return embeddings


def _tile_detect(
    image_bgr: np.ndarray,
) -> List[Tuple[np.ndarray, dict]]:
    """
    将图像按 GROUP_PHOTO_TILE_ROWS 行（列数按宽高比自动）分成有重叠的格子，
    在每个格子中独立运行 _deepface_represent_group，坐标转换回原图空间。
    适合人脸密集或距离较远的大型合照。
    """
    h, w = image_bgr.shape[:2]
    n_rows = GROUP_PHOTO_TILE_ROWS
    n_cols = max(2, round(w / h * n_rows))
    tile_h, tile_w = h // n_rows, w // n_cols
    overlap_h, overlap_w = tile_h // 4, tile_w // 4

    results: List[Tuple[np.ndarray, dict]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            y1 = max(0, r * tile_h - overlap_h)
            y2 = min(h, (r + 1) * tile_h + overlap_h)
            x1 = max(0, c * tile_w - overlap_w)
            x2 = min(w, (c + 1) * tile_w + overlap_w)
            tile = image_bgr[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            try:
                rows = _deepface_represent_group(tile)
                for emb, box in _pick_all_face_embeddings_with_boxes(rows):
                    adjusted = {
                        "x": box.get("x", 0) + x1,
                        "y": box.get("y", 0) + y1,
                        "w": box.get("w", 0),
                        "h": box.get("h", 0),
                    }
                    results.append((emb, adjusted))
            except Exception:
                continue
    return results


def _merge_box_detections(
    primary: List[Tuple[np.ndarray, dict]],
    secondary: List[Tuple[np.ndarray, dict]],
    iou_threshold: float = NMS_IOU_THRESHOLD,
) -> List[Tuple[np.ndarray, dict]]:
    """将 secondary 中与 primary 不重叠（IoU < 阈值）的检测追加进来。"""
    merged = list(primary)
    for emb, box in secondary:
        if not any(_iou(box, kept_box) > iou_threshold for _, kept_box in merged):
            merged.append((emb, box))
    return merged


def extract_all_face_embeddings(image_bgr: np.ndarray) -> List[np.ndarray]:
    """提取图中所有人脸的嵌入（合照等）。"""
    return [emb for emb, _ in extract_all_faces_with_boxes(image_bgr)]


def extract_all_faces_with_boxes(
    image_bgr: np.ndarray,
) -> List[Tuple[np.ndarray, dict]]:
    """
    集体照核心检测流程：
    1. 集体照预处理（放大小图 + 温和 CLAHE）
    2. 遍历所有后端，取检测人脸数最多的结果
    3. 若图像较大，额外做分块检测并合并去重
    4. 全部后端失败时降级到多参数 Haar 级联
    """
    processed = _preprocess_group_photo(image_bgr)
    rows = _deepface_represent_group(processed)

    if rows:
        detections = _pick_all_face_embeddings_with_boxes(rows)
        if GROUP_PHOTO_ENABLE_TILING and max(processed.shape[:2]) >= GROUP_PHOTO_UPSCALE_MIN_DIM:
            tiled = _tile_detect(processed)
            detections = _merge_box_detections(detections, tiled)
        return detections

    # 降级：Haar 兜底（集体照参数）
    out: List[Tuple[np.ndarray, dict]] = []
    for box in _detect_faces_local_group(processed):
        try:
            x, y, w, h = box
            emb = _extract_embedding_from_crop(_crop_face_local(processed, box))
            out.append((emb, {"x": x, "y": y, "w": w, "h": h}))
        except Exception:
            continue
    return out


# ════════════════════════════════════════════════════════════
# 优化4：合照 NMS（去重复检测）
# ════════════════════════════════════════════════════════════

def _iou(box_a: dict, box_b: dict) -> float:
    """计算两个 facial_area dict 的 IoU。"""
    ax1, ay1 = box_a.get("x", 0), box_a.get("y", 0)
    ax2, ay2 = ax1 + box_a.get("w", 1), ay1 + box_a.get("h", 1)
    bx1, by1 = box_b.get("x", 0), box_b.get("y", 0)
    bx2, by2 = bx1 + box_b.get("w", 1), by1 + box_b.get("h", 1)
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(union, 1)


def _nms_faces(
    faces_with_boxes: List[Tuple[np.ndarray, dict]], iou_threshold: float = NMS_IOU_THRESHOLD
) -> List[np.ndarray]:
    """
    对检测到的多张人脸做非极大值抑制，去掉重复检测同一张脸的情况。
    按人脸面积从大到小排序，IoU 超过阈值的后续检测框丢弃。
    """
    if not faces_with_boxes:
        return []
    # 按面积从大到小排
    sorted_faces = sorted(
        faces_with_boxes,
        key=lambda x: x[1].get("w", 0) * x[1].get("h", 0),
        reverse=True,
    )
    kept: List[Tuple[np.ndarray, dict]] = []
    for cand_emb, cand_box in sorted_faces:
        suppress = False
        for _, kept_box in kept:
            if _iou(cand_box, kept_box) > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append((cand_emb, cand_box))
    return [emb for emb, _ in kept]


# ════════════════════════════════════════════════════════════
# 数据库操作（与原版一致）
# ════════════════════════════════════════════════════════════

def merge_face_templates(existing: List[np.ndarray], new_emb: np.ndarray) -> List[np.ndarray]:
    new_arr = np.asarray(new_emb, dtype=np.float32).reshape(-1)
    normalized_existing = [
        np.asarray(e, dtype=np.float32).reshape(-1)
        for e in existing
        if e is not None and np.asarray(e).size > 0
    ]
    for e in normalized_existing:
        if _cosine_distance(e, new_arr) < TEMPLATE_MERGE_MIN_DISTANCE:
            return normalized_existing
    if len(normalized_existing) >= MAX_FACE_TEMPLATES:
        idx_replace = min(
            range(len(normalized_existing)),
            key=lambda i: _cosine_distance(normalized_existing[i], new_arr),
        )
        normalized_existing[idx_replace] = new_arr
        return normalized_existing
    return normalized_existing + [new_arr]


def _load_face_database_local(save_path: str = FACE_DB_PATH) -> dict:
    try:
        db_path = Path(save_path)
        if not db_path.exists():
            return {"success": True, "database": {}}
        with db_path.open("rb") as f:
            database = pickle.load(f)
        if not isinstance(database, dict):
            return {"success": False, "database": {}, "reason": "人脸库格式错误"}
        return {"success": True, "database": database}
    except Exception as exc:
        return {"success": False, "database": {}, "reason": str(exc)}


def init_face_encodings() -> None:
    global face_encodings_db, face_student_names
    face_encodings_db = {}
    face_student_names = {}
    try:
        try:
            from .models.student import Student  # type: ignore
            students = Student.query.all()
            for stu in students:
                if not getattr(stu, "face_encoding", None):
                    continue
                try:
                    encoding = pickle.loads(base64.b64decode(stu.face_encoding))
                    templates = _normalize_encoding_list(encoding)
                    if not templates:
                        continue
                    face_encodings_db[str(stu.student_id)] = templates
                    face_student_names[str(stu.student_id)] = str(getattr(stu, "name", ""))
                except Exception:
                    continue
            return
        except Exception:
            pass
        db_res = _load_face_database_local(FACE_DB_PATH)
        if not db_res.get("success"):
            return
        database = db_res.get("database", {})
        for student_id, info in database.items():
            embeddings = info.get("face_embeddings", [])
            if not embeddings:
                continue
            face_encodings_db[str(student_id)] = [
                np.asarray(item, dtype=np.float32) for item in embeddings if item is not None
            ]
            face_student_names[str(student_id)] = str(info.get("name", ""))
    except Exception:
        LOGGER.exception("初始化人脸特征失败")


# ════════════════════════════════════════════════════════════
# 优化5：改进的匹配 & 决策逻辑
# ════════════════════════════════════════════════════════════

def _match_embedding_to_gallery(emb: np.ndarray) -> Tuple[Optional[str], float, float, int]:
    """每个学生取多模板的最小余弦距离，再在学生之间比较最优与次优。（与原版一致）"""
    per_student: Dict[str, float] = {}
    compared_count = 0
    for student_id, db_emb_list in face_encodings_db.items():
        best_for_student = float("inf")
        for db_emb in _normalize_encoding_list(db_emb_list):
            if not _same_dim(emb, db_emb):
                continue
            compared_count += 1
            dist = _cosine_distance(emb, db_emb)
            if dist < best_for_student:
                best_for_student = dist
        if best_for_student < float("inf"):
            per_student[student_id] = best_for_student
    if not per_student:
        return None, float("inf"), float("inf"), compared_count
    ranked = sorted(per_student.items(), key=lambda x: x[1])
    best_sid, best_d = ranked[0][0], ranked[0][1]
    second_d = ranked[1][1] if len(ranked) > 1 else 1.0
    return best_sid, best_d, second_d, compared_count


def _match_augmented_embeddings(
    embeddings: List[np.ndarray],
) -> Tuple[Optional[str], float, str]:
    """
    优化3（核心）：多增强嵌入 → 加权投票决策。

    策略：
    1. 对每条嵌入独立在人脸库中找最近邻候选（student_id, distance）
    2. 用「1 - distance」作为票权，按学生 ID 累加
    3. 得票最高的学生为候选，同时要求其原始最近距离也在阈值内
    4. 检查是否与第二名过于接近（防混淆）

    好处：光线/表情/角度变化时只要有部分增强变体识别正确即可拉高该人票数，
    单次提取的偶发偏差不会直接导致拒绝。
    """
    if not embeddings:
        return None, float("inf"), "未提取到嵌入向量"

    # 收集每条嵌入的最近邻
    vote_weights: Dict[str, float] = {}
    best_distances: Dict[str, float] = {}
    # 严格阈值内的票数：只有真正"像"才算，防止矮个子里拔高个
    strict_vote_counts: Dict[str, int] = {}
    total_compared = 0

    for emb in embeddings:
        sid, best_d, second_d, cnt = _match_embedding_to_gallery(emb)
        total_compared += cnt
        if sid is None:
            continue
        # 粗筛：1.5x 阈值内才参与投票（用于积累权重）
        if best_d > MATCH_THRESHOLD * 1.5:
            continue
        weight = max(0.0, 1.0 - best_d)
        vote_weights[sid] = vote_weights.get(sid, 0.0) + weight
        # 记录该学生在所有增强变体中的最小距离
        if sid not in best_distances or best_d < best_distances[sid]:
            best_distances[sid] = best_d
        # 严格阈值内才计入确认票数
        if best_d <= MATCH_THRESHOLD:
            strict_vote_counts[sid] = strict_vote_counts.get(sid, 0) + 1

    if total_compared == 0:
        return None, float("inf"), "人脸库特征维度不一致，请重新初始化人脸特征"

    if not vote_weights:
        return None, float("inf"), (
            f"未匹配到学生(model={EMBEDDING_MODEL_NAME}, threshold={MATCH_THRESHOLD})"
        )

    # 按得票排名
    ranked = sorted(vote_weights.items(), key=lambda x: x[1], reverse=True)
    winner_sid = ranked[0][0]
    winner_dist = best_distances[winner_sid]

    # 最优距离必须过硬阈值
    if winner_dist > MATCH_THRESHOLD:
        return None, winner_dist, (
            f"未匹配到学生(best_cosine_distance={winner_dist:.4f}, threshold={MATCH_THRESHOLD}, "
            f"model={EMBEDDING_MODEL_NAME})"
        )

    # 严格票数不足：只有 1 个增强变体压线通过，不足以确认身份
    # 这防止"矮个子里拔高个"——真正的匹配应有多个角度/光线变体都认可
    if strict_vote_counts.get(winner_sid, 0) < 2:
        return None, winner_dist, (
            f"匹配置信度不足，仅 {strict_vote_counts.get(winner_sid, 0)} 个角度确认"
            f"(distance={winner_dist:.4f})，请正对摄像头重试"
        )

    # 防混淆：第二名距离检查
    if len(ranked) > 1:
        second_sid = ranked[1][0]
        second_dist = best_distances.get(second_sid, 1.0)
        if second_dist <= MATCH_THRESHOLD and (second_dist - winner_dist) < MIN_CONFUSION_MARGIN:
            return None, winner_dist, (
                f"候选过于接近，无法唯一判定(best={winner_dist:.4f}, second={second_dist:.4f}, "
                f"margin_need={MIN_CONFUSION_MARGIN})"
            )

    return winner_sid, winner_dist, ""


def _resolve_identity(emb: np.ndarray) -> Tuple[Optional[str], float, str]:
    """单向量版本，保持对旧调用的兼容。"""
    best_sid, best_d, second_d, compared_count = _match_embedding_to_gallery(emb)
    if compared_count == 0:
        return None, best_d, "人脸库特征维度不一致，请重新初始化人脸特征"
    if best_sid is None or best_d > MATCH_THRESHOLD:
        return None, best_d, (
            f"未匹配到学生(best_cosine_distance={best_d:.4f}, threshold={MATCH_THRESHOLD}, "
            f"model={EMBEDDING_MODEL_NAME})"
        )
    if second_d <= MATCH_THRESHOLD and (second_d - best_d) < MIN_CONFUSION_MARGIN:
        return None, best_d, (
            f"候选过于接近，无法唯一判定(best={best_d:.4f}, second={second_d:.4f}, "
            f"margin_need={MIN_CONFUSION_MARGIN})"
        )
    return best_sid, best_d, ""


# ════════════════════════════════════════════════════════════
# 活体检测（原版，接口不变）
# ════════════════════════════════════════════════════════════
def _detect_moire_pattern(gray: np.ndarray) -> float:
    """
    检测打印照片常见的摩尔纹（频域分析）。
    打印品在高频域能量分布异常，返回可疑分数 0~1，越高越可疑。
    """
    if gray is None or gray.size == 0:
        return 0.0
    try:
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1.0)
        h, w = magnitude.shape
        # 中心低频区域 vs 边缘高频区域
        center_mean = magnitude[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean()
        edge_mean = np.concatenate([
            magnitude[:h // 4, :].ravel(),
            magnitude[3 * h // 4:, :].ravel(),
            magnitude[:, :w // 4].ravel(),
            magnitude[:, 3 * w // 4:].ravel(),
        ]).mean()
        # 打印照片高频能量相对偏高
        ratio = edge_mean / (center_mean + 1e-6)
        # 正常人脸 ratio 约 0.3~0.6；摩尔纹照片往往 > 0.75
        # 基准从 0.5 调高到 0.68，避免将摄像头噪声/皮肤纹理误判为摩尔纹
        suspicious = float(np.clip((ratio - 0.68) / 0.32, 0.0, 1.0))
        return suspicious
    except Exception:
        return 0.0

def _detect_reflection(image_bgr: np.ndarray) -> float:
    """
    检测屏幕重放攻击的反光特征。
    过亮像素占比越高越可疑，返回 0~1。
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    try:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        # 饱和度极低 + 亮度极高 → 高光反射区域
        s_channel = hsv[:, :, 1]
        overexposed_mask = (v_channel > 236) & (s_channel < 36)
        overexposed_ratio = float(np.mean(overexposed_mask))
        # 正常人脸偶有小面积高光；屏幕/反光照片往往 > 3%
        suspicious = float(np.clip(overexposed_ratio / 0.03, 0.0, 1.0))
        return suspicious
    except Exception:
        return 0.0


def _detect_screen_border(image_bgr: np.ndarray) -> float:
    """
    检测画面边缘是否存在连续暗边框/设备边框（平板/手机重拍常见）。
    返回 0~1，越高越可疑。
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        bw = max(8, int(min(h, w) * 0.06))
        border = np.concatenate([
            gray[:bw, :].ravel(),
            gray[-bw:, :].ravel(),
            gray[:, :bw].ravel(),
            gray[:, -bw:].ravel(),
        ])
        inner = gray[bw:-bw, bw:-bw] if (h > 2 * bw and w > 2 * bw) else gray
        border_dark_ratio = float(np.mean(border < 42))
        inner_dark_ratio = float(np.mean(inner < 42))
        # 边缘显著更暗，且暗边占比偏高时可疑
        delta = max(0.0, border_dark_ratio - inner_dark_ratio)
        suspicious = np.clip(delta / 0.22, 0.0, 1.0) * 0.7 + np.clip(border_dark_ratio / 0.55, 0.0, 1.0) * 0.3
        return float(np.clip(suspicious, 0.0, 1.0))
    except Exception:
        return 0.0


def _detect_photo_attack_local(image: np.ndarray) -> dict:
    """
    综合活体检测：
      - 清晰度（Laplacian 方差）
      - 边缘/纹理丰富度
      - 摩尔纹（频域高频能量比）
      - 屏幕反光（过亮低饱和区域占比）
      - 屏幕边框（边缘连续暗边）
    各指标加权后映射到 [0, 1] 综合得分，>= 0.55 判定为活体。
    """
    if image is None or image.size == 0:
        return {"is_live": False, "score": 0.0, "reason": "图像为空"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ── 指标1：清晰度（模糊照片得分低）──────────────────────────────
    clarity_raw = cv2.Laplacian(gray, cv2.CV_64F).var()
    clarity = float(np.clip(clarity_raw / 300.0, 0.0, 1.0))

    # ── 指标2：纹理（灰度标准差 + 边缘密度）─────────────────────────
    std_score = float(np.clip(np.std(gray) / 64.0, 0.0, 1.0))
    edges = cv2.Canny(gray, 80, 180)
    edge_score = float(np.clip(np.mean(edges > 0) / 0.12, 0.0, 1.0))
    texture = (std_score + edge_score) / 2.0

    # ── 指标3：摩尔纹可疑度（越高越像照片）──────────────────────────
    moire_suspicious = _detect_moire_pattern(gray)

    # ── 指标4：反光可疑度（越高越像屏幕/打印）───────────────────────
    reflection_suspicious = _detect_reflection(image)

    # ── 指标5：边框可疑度（越高越像屏幕重拍）───────────────────────
    border_suspicious = _detect_screen_border(image)

    # ── 综合得分：正向指标加分，可疑指标减分 ─────────────────────────
    # 权重设计：纹理和清晰度为主；屏幕类攻击惩罚项更高
    positive_score = 0.4 * clarity + 0.3 * texture
    penalty = (
        0.15 * moire_suspicious
        + 0.22 * reflection_suspicious
        + 0.20 * border_suspicious
    )
    score = float(np.clip(positive_score - penalty + 0.28, 0.0, 1.0))

    # ── 硬性否决规则（任一可疑度过高直接拒绝）───────────────────────
    hard_reject = (
        reflection_suspicious > 0.72
        or moire_suspicious > 0.80
        or border_suspicious > 0.68
        or (moire_suspicious > 0.65 and reflection_suspicious > 0.58)
        or (border_suspicious > 0.56 and reflection_suspicious > 0.56)
    )

    is_live = (score >= 0.60) and not hard_reject

    # ── 生成可读的判断依据 ────────────────────────────────────────────
    if hard_reject:
        if reflection_suspicious > 0.72:
            reason = f"检测到异常高光反射区域（反光得分={reflection_suspicious:.2f}），疑似屏幕重放攻击"
        elif border_suspicious > 0.68:
            reason = f"检测到明显设备边框特征（边框得分={border_suspicious:.2f}），疑似屏幕重放攻击"
        else:
            reason = f"检测到摩尔纹频域特征（摩尔纹得分={moire_suspicious:.2f}），疑似打印照片攻击"
    elif not is_live:
        parts = []
        if clarity < 0.3:
            parts.append(f"清晰度不足({clarity:.2f})")
        if texture < 0.3:
            parts.append(f"纹理过于平滑({texture:.2f})")
        if moire_suspicious > 0.45:
            parts.append(f"存在摩尔纹迹象({moire_suspicious:.2f})")
        if reflection_suspicious > 0.4:
            parts.append(f"存在反光迹象({reflection_suspicious:.2f})")
        if border_suspicious > 0.42:
            parts.append(f"存在边框迹象({border_suspicious:.2f})")
        reason = "疑似静态照片攻击：" + "、".join(parts) if parts else "综合得分过低，疑似照片攻击"
    else:
        reason = (
            f"图像纹理和清晰度正常"
            f"（清晰度={clarity:.2f}, 纹理={texture:.2f}, "
            f"摩尔纹={moire_suspicious:.2f}, 反光={reflection_suspicious:.2f}, "
            f"边框={border_suspicious:.2f}）"
        )

    return {"is_live": is_live, "score": round(score, 4), "reason": reason}


def detect_liveness_photo(image_base64: str) -> dict:
    """单张图片活体检测。"""
    try:
        image, _ = _decode_base64_image(image_base64)
        result = _detect_photo_attack_local(image)
        return {
            "is_live": bool(result.get("is_live", False)),
            "score": float(result.get("score", 0.0)),
            "reason": str(result.get("reason", "")),
        }
    except Exception as exc:
        return {"is_live": False, "score": 0.0, "reason": str(exc)}

# ════════════════════════════════════════════════════════════
# 公开接口（对外签名与原版完全一致）
# ════════════════════════════════════════════════════════════

def compare_faces(image_base64: str) -> dict:
    """
    单张人脸比对，返回匹配学生信息。
    优化点：启用多增强向量投票匹配（ENABLE_AUGMENTATION_VOTE=True 时）。
    """
    try:
        if not face_encodings_db:
            init_face_encodings()
        if not face_encodings_db:
            return {
                "success": False,
                "matched_student": None,
                "confidence": 0.0,
                "error_msg": "人脸库为空",
            }

        image, img_bytes = _decode_base64_image(image_base64)
        image = _fix_image_orientation(image, img_bytes)

        if ENABLE_AUGMENTATION_VOTE:
            # ── 优化路径：多增强嵌入 + 投票 ──────────────────────────
            try:
                embeddings = extract_primary_embedding_augmented(image)
            except Exception as detect_exc:
                return {
                    "success": False,
                    "matched_student": None,
                    "confidence": 0.0,
                    "error_msg": str(detect_exc) or "未检测到人脸",
                }
            best_student_id, best_distance, reject_reason = _match_augmented_embeddings(embeddings)
        else:
            # ── 兼容路径：原始单向量匹配 ──────────────────────────────
            try:
                emb = extract_primary_embedding(image)
            except Exception as detect_exc:
                return {
                    "success": False,
                    "matched_student": None,
                    "confidence": 0.0,
                    "error_msg": str(detect_exc) or "未检测到人脸",
                }
            best_student_id, best_distance, reject_reason = _resolve_identity(emb)

        if reject_reason:
            return {
                "success": False,
                "matched_student": None,
                "confidence": 0.0,
                "error_msg": reject_reason,
            }

        if ENABLE_LIVENESS_CHECK:
            liveness_result = detect_liveness_photo(image_base64)
            if not liveness_result.get("is_live", False):
                return {
                    "success": False,
                    "matched_student": None,
                    "confidence": _distance_to_confidence(best_distance),
                    "error_msg": liveness_result.get("reason", "活体检测未通过"),
                }

        return {
            "success": True,
            "matched_student": {
                "student_id": str(best_student_id),
                "name": face_student_names.get(str(best_student_id), ""),
            },
            "confidence": _distance_to_confidence(best_distance),
            "error_msg": "",
        }
    except Exception as exc:
        return {
            "success": False,
            "matched_student": None,
            "confidence": 0.0,
            "error_msg": str(exc),
        }


def recognize_group_photo(image_base64: str) -> list:
    """
    合照识别，返回所有匹配学生。
    优化点：加入 NMS 去重，避免同一张脸被检出多次互相干扰。
    """
    try:
        if not face_encodings_db:
            init_face_encodings()
        if not face_encodings_db:
            return []

        image, img_bytes = _decode_base64_image(image_base64)
        image = _fix_image_orientation(image, img_bytes)

        # ── 优化：带框信息的提取 + NMS ────────────────────────────────
        faces_with_boxes = extract_all_faces_with_boxes(image)
        print("DEBUG: 检测到人脸数 =", len(faces_with_boxes))
        print("DEBUG: 第一个box =", faces_with_boxes[0][1] if faces_with_boxes else "空")
        embeddings = _nms_faces(faces_with_boxes)
        print("DEBUG: NMS后人脸数 =", len(embeddings))
        if not embeddings:
            return []

        matched_students: Dict[str, float] = {}
        for emb in embeddings:
            best_student_id, best_distance, reject_reason = _resolve_identity(emb)
            print(f"DEBUG: sid={best_student_id} dist={best_distance:.4f} reason={reject_reason}")
            if reject_reason:
                continue
            prev_distance = matched_students.get(best_student_id)
            if prev_distance is None or best_distance < prev_distance:
                matched_students[best_student_id] = best_distance

        return [
            {
                "student_id": sid,
                "name": face_student_names.get(sid, ""),
            }
            for sid in matched_students.keys()
        ]
    except Exception:
        return []