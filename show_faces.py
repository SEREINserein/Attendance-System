"""查看数据库中已录入人脸的管理脚本。

用法：
  python backend/show_faces.py            # 显示所有学生
  python backend/show_faces.py 2024002    # 只显示指定学号
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches

# ── 查找 Windows 中文字体 ──────────────────────────────────────────
def _find_chinese_font():
    candidates = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return fm.FontProperties(fname=p)
    return fm.FontProperties()  # 找不到时用默认字体

CN_FONT = _find_chinese_font()

def cn(text, size=10, **kwargs):
    """返回带中文字体的 Text 参数字典。"""
    return dict(fontproperties=CN_FONT, fontsize=size, **kwargs)


# ── 主逻辑 ────────────────────────────────────────────────────────
from backend.app import create_app
from backend.models.student import Student

FACE_PHOTO_DIR = PROJECT_ROOT / "backend" / "data" / "face_photos"

app = create_app()
target_ids = sys.argv[1:] if len(sys.argv) > 1 else None

with app.app_context():
    students = Student.query.all()
    if target_ids:
        students = [s for s in students if str(s.student_id) in target_ids]

    entries = []
    for stu in students:
        sid = str(stu.student_id)
        name = getattr(stu, "name", "")
        has_encoding = bool(getattr(stu, "face_encoding", None))
        photo_path = FACE_PHOTO_DIR / f"{sid}.jpg"
        img = None
        if photo_path.exists() and has_encoding:
            raw = cv2.imread(str(photo_path))
            if raw is not None:
                img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        entries.append((sid, name, has_encoding, img, photo_path.exists()))

if not entries:
    print("未找到符合条件的学生。")
    sys.exit(0)

n = len(entries)
cols = min(n, 4)
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 4.5))
fig.suptitle("数据库已录入人脸", **cn(14))

if n == 1:
    axes = np.array([[axes]])
elif rows == 1:
    axes = np.array([axes])

for idx, (sid, name, has_encoding, img, photo_exists) in enumerate(entries):
    ax = axes[idx // cols][idx % cols]
    ax.axis("off")

    if img is not None:
        # 正常状态：有特征 + 有照片
        ax.imshow(img)
        status_color = "#2ecc71"
        status_text = "✓ 已录入特征"
    else:
        # 灰色占位图
        placeholder = np.full((240, 200, 3), 230, dtype=np.uint8)
        # 画一个人形轮廓示意
        cv2.circle(placeholder, (100, 80), 40, (180, 180, 180), -1)
        cv2.ellipse(placeholder, (100, 190), (60, 50), 0, 0, 180, (180, 180, 180), -1)
        ax.imshow(placeholder)

        if not has_encoding and photo_exists:
            status_color = "#e74c3c"
            status_text = "✗ 人脸数据已删除"
        elif not has_encoding:
            status_color = "#e74c3c"
            status_text = "✗ 未录入人脸"
        else:
            # has_encoding but no photo file (旧数据注册时未保存照片)
            status_color = "#f39c12"
            status_text = "⚠ 有特征但无照片"

    ax.set_title(f"{sid}  {name}", **cn(10))
    patch = mpatches.Patch(color=status_color, label=status_text)
    ax.legend(handles=[patch], loc="lower center",
              prop=CN_FONT, fontsize=8,
              bbox_to_anchor=(0.5, -0.02), frameon=True)

for idx in range(n, rows * cols):
    axes[idx // cols][idx % cols].set_visible(False)

plt.tight_layout()
out_path = PROJECT_ROOT / "backend" / "data" / "face_photos_preview.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
print(f"预览图已保存: {out_path}")

try:
    plt.show()
except Exception:
    pass
