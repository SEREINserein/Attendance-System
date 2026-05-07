"""诊断：对所有测试图片与所有DB模板计算距离矩阵，找出污染来源。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import base64, pickle
import cv2
import numpy as np

from backend.app import create_app
from backend.database.db import db
from backend.models.student import Student
from backend.face_recognition import (
    extract_primary_embedding,
    _cosine_distance,
    _normalize_encoding_list,
    MATCH_THRESHOLD,
)

app = create_app()
TEST_DIR = PROJECT_ROOT / "backend" / "test_images"

with app.app_context():
    # 读取DB中的模板
    gallery = {}
    names = {}
    for stu in Student.query.all():
        sid = str(stu.student_id)
        names[sid] = getattr(stu, "name", "")
        enc = getattr(stu, "face_encoding", None)
        if enc:
            decoded = pickle.loads(base64.b64decode(enc))
            gallery[sid] = _normalize_encoding_list(decoded)
        else:
            gallery[sid] = []

    print("=" * 70)
    print("各测试图片 → 各学生DB模板的最小余弦距离")
    print(f"（阈值={MATCH_THRESHOLD}，低于阈值=匹配）")
    print("=" * 70)
    sids = sorted(gallery.keys())
    header = f"{'图片文件':<20}" + "".join(f"{sid}({names[sid]}){'':<3}" for sid in sids)
    print(header)
    print("-" * 70)

    for img_path in sorted(TEST_DIR.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        try:
            emb = extract_primary_embedding(img)
        except Exception as e:
            print(f"{img_path.name:<20}  提取失败: {e}")
            continue

        row = f"{img_path.name:<20}"
        best_sid, best_d = None, float("inf")
        for sid in sids:
            if not gallery[sid]:
                row += f"  {'N/A':>8}  "
                continue
            dists = [_cosine_distance(emb, np.asarray(t, dtype=np.float32)) for t in gallery[sid]]
            d = min(dists)
            mark = " *" if d <= MATCH_THRESHOLD else "  "
            row += f"  {d:>8.4f}{mark}"
            if d < best_d:
                best_d, best_sid = d, sid
        row += f"  → 最近: {best_sid}({names.get(best_sid, '')})"
        print(row)

    print()
    print("=" * 70)
    print("stu003 各模板 与 所有测试图片嵌入 的距离")
    print("（检查是否有女脸混入stu003模板库）")
    print("=" * 70)
    stu003_templates = gallery.get("2024003", [])
    for ti, tmpl in enumerate(stu003_templates):
        tmpl_arr = np.asarray(tmpl, dtype=np.float32)
        print(f"\nstu003 模板{ti} (std={tmpl_arr.std():.4f}):")
        for img_path in sorted(TEST_DIR.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            try:
                emb = extract_primary_embedding(img)
                d = _cosine_distance(emb, tmpl_arr)
                flag = " ← 疑似来源！" if d < 0.15 else (" ← 阈值内" if d <= MATCH_THRESHOLD else "")
                print(f"  {img_path.name:<20} 距离={d:.4f}{flag}")
            except Exception:
                pass
