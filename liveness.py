"""
liveness.py — 活体检测模块
================================================================
版本：v6  修复批量帧检测失效 & 多线程安全

修复说明：
  - static_image_mode 改为 True：WebSocket 是批量收帧后集中处理，
    不是实时流，False 模式下追踪器无法建立，导致全部帧检测失败。
  - 加 threading.Lock：WebSocket 服务跑在独立线程，防止并发调用
    同一 FaceMesh 实例造成结果错乱。

依赖：
    pip install opencv-python mediapipe==0.10.9 numpy

公开接口：
    DISPLAY_TEXT                            → 动作中文提示字典
    get_challenge_sequence(n=3) -> list     → 生成随机挑战序列
    detect_liveness_video(frames_base64, required_action) -> dict
"""

import base64
import random
import threading
import cv2
import numpy as np
import mediapipe as mp

# ─────────────────────────────────────────
# 检测参数（内部常量，以 _ 开头）
# ─────────────────────────────────────────

_EAR_THRESHOLD  = 0.22   # EAR 低于此值 → 闭眼
_MAR_THRESHOLD  = 0.55   # MAR 高于此值 → 张嘴
_TURN_THRESHOLD = 0.07   # 鼻尖水平偏移比例高于此值 → 转头
_NOD_THRESHOLD  = 0.04   # 鼻尖垂直位移范围高于此值 → 点头
_BLINK_CONSEC   = 2      # 连续闭眼帧数 >= 此值才算一次眨眼
_MIN_FRAMES     = 5      # 有效帧数下限

# ─────────────────────────────────────────
# 公开常量：动作 → 中文提示（B 同学前端显示用）
# 注意：此字典的键名和值均不可更改
# ─────────────────────────────────────────

DISPLAY_TEXT = {
    "blink":      "请眨眼",
    "mouth":      "请张嘴",
    "turn_left":  "请向左转头",
    "turn_right": "请向右转头",
    "nod":        "请点头",
}

# ─────────────────────────────────────────
# MediaPipe 初始化（模块级单例 + 线程锁）
# ─────────────────────────────────────────

# static_image_mode=True：每帧独立检测，不依赖帧间追踪。
# WebSocket 场景下帧是批量收集后集中处理的，不是连续视频流，
# False 模式下追踪器永远无法建立，会导致所有帧检测失败。
_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)
_face_mesh_lock = threading.Lock()  # WebSocket 服务跑在独立线程，防并发

# 关键点索引（MediaPipe Face Mesh 478 点）
_LEFT_EYE    = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE   = [33,  160, 158, 133, 153, 144]
_MOUTH       = [61, 291, 13, 14]   # 左, 右, 上唇, 下唇
_NOSE_TIP    = 1
_LEFT_CHEEK  = 234
_RIGHT_CHEEK = 454
_CHIN        = 152   # 下巴（用于脸部高度归一化）
_FOREHEAD    = 10    # 额头


# ─────────────────────────────────────────
# 内部辅助函数
# ─────────────────────────────────────────

def _strip_data_url(b64_str: str) -> str:
    """去除 data:image/...;base64, 前缀，返回纯 base64 字符串。"""
    if "," in b64_str:
        return b64_str.split(",", 1)[1]
    return b64_str


def _decode_frame(b64_str: str):
    """base64 字符串 → OpenCV BGR 帧；失败返回 None。"""
    try:
        raw = base64.b64decode(_strip_data_url(b64_str))
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _eye_aspect_ratio(lm, indices):
    """眼睛纵横比（EAR）：值越小眼睛越闭合。"""
    pts = np.array([[lm[i].x, lm[i].y] for i in indices])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h  = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def _mouth_aspect_ratio(lm):
    """嘴巴纵横比（MAR）：值越大嘴张得越开。"""
    left  = np.array([lm[_MOUTH[0]].x, lm[_MOUTH[0]].y])
    right = np.array([lm[_MOUTH[1]].x, lm[_MOUTH[1]].y])
    top   = np.array([lm[_MOUTH[2]].x, lm[_MOUTH[2]].y])
    bot   = np.array([lm[_MOUTH[3]].x, lm[_MOUTH[3]].y])
    return np.linalg.norm(top - bot) / (np.linalg.norm(left - right) + 1e-6)


def _head_turn_ratio(lm):
    """
    鼻尖相对脸部中轴的水平偏移比例。
    正值 → 鼻尖偏右（用户向左转）；负值 → 鼻尖偏左（用户向右转）。
    """
    nose = lm[_NOSE_TIP].x
    l, r = lm[_LEFT_CHEEK].x, lm[_RIGHT_CHEEK].x
    return (nose - (l + r) / 2.0) / (r - l + 1e-6)


def _nose_y_normalized(lm) -> float:
    """
    鼻尖在脸部高度上的归一化 Y 位置（额头=0，下巴=1）。
    用于跨帧检测点头。
    """
    nose_y     = lm[_NOSE_TIP].y
    forehead_y = lm[_FOREHEAD].y
    chin_y     = lm[_CHIN].y
    return (nose_y - forehead_y) / (chin_y - forehead_y + 1e-6)


def _detect_nod(nose_y_series: list) -> bool:
    """
    判断归一化鼻尖 Y 序列是否包含点头动作。
    条件：
      1. 整体 Y 变化幅度 > _NOD_THRESHOLD
      2. 序列存在方向反转（下→上 或 上→下），排除单方向倾斜
    """
    if len(nose_y_series) < _MIN_FRAMES:
        return False
    arr = np.array(nose_y_series)
    if arr.max() - arr.min() < _NOD_THRESHOLD:
        return False
    diffs = np.diff(arr)
    signs = np.sign(diffs[diffs != 0])
    if len(signs) < 2:
        return False
    return bool((signs[1:] != signs[:-1]).any())


# ─────────────────────────────────────────
# 公开接口 1
# ─────────────────────────────────────────

def get_challenge_sequence(n: int = 3) -> list:
    """
    生成 n 个不重复的随机动作挑战序列。

    参数：
        n: 挑战数量，默认 3

    返回：
        list[str]，元素为 DISPLAY_TEXT 中的键，例如 ['blink', 'mouth', 'turn_left']
    """
    actions = list(DISPLAY_TEXT.keys())
    return random.sample(actions, min(n, len(actions)))


# ─────────────────────────────────────────
# 公开接口 2
# ─────────────────────────────────────────

def detect_liveness_video(frames_base64: list, required_action: str) -> dict:
    """
    分析连续视频帧，判断用户是否完成了指定动作。

    参数：
        frames_base64:   list[str]，base64 编码的帧
                         （自动处理 data:image/...;base64, 前缀）
        required_action: 指定动作，取值为 DISPLAY_TEXT 中的键

    返回：
        {
            "action_completed": bool,   # 动作是否完成
            "reason":           str,    # 完成时为空字符串，失败时说明原因
        }

    保证：内部全程 try-except，不会向外抛出未捕获异常。
    """
    def _fail(msg: str) -> dict:
        return {"action_completed": False, "reason": msg}

    try:
        # ── 参数校验 ──
        if required_action not in DISPLAY_TEXT:
            return _fail(f"未知动作类型: {required_action}")

        if not frames_base64 or len(frames_base64) < _MIN_FRAMES:
            return _fail(
                f"帧数不足（收到 {len(frames_base64) if frames_base64 else 0} 帧，"
                f"需要至少 {_MIN_FRAMES} 帧）"
            )

        # ── 逐帧分析 ──
        blink_counter   = 0
        blink_detected  = False
        mouth_detected  = False
        turn_l_detected = False
        turn_r_detected = False
        nose_y_series   = []
        valid_frames    = 0

        for b64 in frames_base64:
            try:
                frame = _decode_frame(b64)
                if frame is None:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with _face_mesh_lock:
                    results = _face_mesh.process(rgb)
                if not results.multi_face_landmarks:
                    continue

                lm = results.multi_face_landmarks[0].landmark
                valid_frames += 1

                # 眨眼
                ear = (
                    _eye_aspect_ratio(lm, _LEFT_EYE)
                    + _eye_aspect_ratio(lm, _RIGHT_EYE)
                ) / 2.0
                if ear < _EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= _BLINK_CONSEC:
                        blink_detected = True
                    blink_counter = 0

                # 张嘴
                if _mouth_aspect_ratio(lm) > _MAR_THRESHOLD:
                    mouth_detected = True

                # 转头
                turn = _head_turn_ratio(lm)
                if turn >  _TURN_THRESHOLD:
                    turn_l_detected = True
                if turn < -_TURN_THRESHOLD:
                    turn_r_detected = True

                # 点头（收集 Y 序列，循环后统一判断）
                nose_y_series.append(_nose_y_normalized(lm))

            except Exception:
                continue   # 单帧异常不中断整体

        # ── 无有效人脸 ──
        if valid_frames == 0:
            return _fail("没有检测到有效人脸，请确保面部正对摄像头")

        # 点头：基于完整序列判断
        nod_detected = _detect_nod(nose_y_series)

        # ── 根据 required_action 判断结果 ──
        action_map = {
            "blink":      blink_detected,
            "mouth":      mouth_detected,
            "turn_left":  turn_l_detected,
            "turn_right": turn_r_detected,
            "nod":        nod_detected,
        }

        completed = action_map[required_action]
        reason = (
            ""
            if completed
            else (
                f"未检测到【{DISPLAY_TEXT[required_action]}】，"
                "请确保动作幅度足够且面部正对摄像头"
            )
        )
        return {"action_completed": completed, "reason": reason}

    except Exception as e:
        return _fail(f"检测过程发生内部错误: {e}")