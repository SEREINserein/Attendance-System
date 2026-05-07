"""
人脸特征录入：支持单张或批量。

单张示例:
  python backend/register_face_once.py --student-id 2024001 --image path/to/photo.jpg

批量示例（stu001.jpg → 学号 2024001，前缀与数字位数可配）:
  python backend/register_face_once.py --bulk backend/test_images --pattern stu --id-prefix 2024 --suffix-width 3

批量示例（文件名即学号，如 2024001.jpg）:
  python backend/register_face_once.py --bulk backend/test_images --pattern direct

兼容旧用法（未传参数时仍读取下方 STUDENT_ID / IMAGE_PATH）:
  python backend/register_face_once.py
"""

from __future__ import annotations

import argparse
import base64
import pickle
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# 允许直接运行: python backend/register_face_once.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import create_app
from backend.database.db import db
from backend.models.student import Student
from backend.face_recognition import extract_primary_embedding, init_face_encodings, merge_face_templates

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ------- 无命令行参数时的默认单张录入（兼容旧脚本）-------
STUDENT_ID = "2024001"
IMAGE_PATH = PROJECT_ROOT / "backend" / "test_images" / "stu001.png"


def stem_to_student_id(
    stem: str,
    pattern: str,
    *,
    id_prefix: str,
    suffix_width: int,
) -> str | None:
    """根据文件名（不含扩展名）解析学号。"""
    if pattern == "direct":
        s = stem.strip()
        if not s:
            return None
        # 常见学号：纯数字或以年份开头
        if s.isdigit() or (len(s) >= 4 and s[:4].isdigit() and s[4:].isdigit()):
            return s
        return None

    if pattern == "stu":
        m = re.match(r"^stu(\d+)$", stem.strip(), flags=re.IGNORECASE)
        if not m:
            return None
        num = int(m.group(1))
        return f"{id_prefix}{num:0{suffix_width}d}"

    raise ValueError(f"未知 pattern: {pattern}")


def iter_image_files(folder: Path):
    if not folder.is_dir():
        return
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def load_existing_templates(stu: Student) -> list[np.ndarray]:
    existing: list[np.ndarray] = []
    if not stu.face_encoding:
        return existing
    try:
        prev = pickle.loads(base64.b64decode(stu.face_encoding))
        if isinstance(prev, list):
            existing = [np.asarray(x, dtype=np.float32) for x in prev if x is not None]
        elif prev is not None:
            existing = [np.asarray(prev, dtype=np.float32)]
    except Exception:
        pass
    return existing


FACE_PHOTO_DIR = PROJECT_ROOT / "backend" / "data" / "face_photos"


def _save_face_photo(student_id: str, image: np.ndarray) -> None:
    """将注册时使用的照片保存到 face_photos 目录，供后续查看。"""
    FACE_PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FACE_PHOTO_DIR / f"{student_id}.jpg"
    cv2.imwrite(str(out_path), image)


def register_one_image(student_id: str, image_path: Path) -> tuple[bool, str]:
    """将一张照片并入该学号的模板库，提交数据库。"""
    stu = Student.query.get(student_id)
    if not stu:
        return False, f"学号不存在: {student_id}"

    image = cv2.imread(str(image_path))
    if image is None:
        return False, "图片解码失败"

    try:
        emb = extract_primary_embedding(image)
    except Exception as exc:
        return False, f"提取人脸特征失败: {exc}"

    existing = load_existing_templates(stu)
    merged = merge_face_templates(existing, emb)
    stu.face_encoding = base64.b64encode(pickle.dumps(merged)).decode("utf-8")
    db.session.commit()

    _save_face_photo(student_id, image)

    return True, f"模板数={len(merged)} 向量维度={int(np.asarray(merged[0]).size)}"


def parse_args() -> argparse.Namespace | None:
    parser = argparse.ArgumentParser(description="人脸特征录入（单张或批量）")
    parser.add_argument("--bulk", type=Path, metavar="DIR", help="批量模式：图片目录")
    parser.add_argument(
        "--pattern",
        choices=("stu", "direct"),
        default="stu",
        help="文件名解析：stu → stu001→前缀+序号；direct → 2024001.jpg 即用 stem 作学号",
    )
    parser.add_argument(
        "--id-prefix",
        default="2024",
        help="stu 模式下学号前缀（默认 2024，stu001→2024001）",
    )
    parser.add_argument(
        "--suffix-width",
        type=int,
        default=3,
        help="stu 模式下序号零填充宽度（默认 3）",
    )
    parser.add_argument("--student-id", metavar="ID", help="单张模式：学号")
    parser.add_argument("--image", type=Path, metavar="PATH", help="单张模式：图片路径")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的映射，不写数据库",
    )
    args = parser.parse_args()
    if args.bulk is None and args.student_id is None and args.image is None:
        return None
    return args


def run_bulk(args: argparse.Namespace) -> None:
    folder = args.bulk.resolve()
    dry = args.dry_run
    ok_count = 0
    skip: list[tuple[str, str]] = []
    errors: list[tuple[str, str]] = []

    files = list(iter_image_files(folder))
    if not files:
        print(f"目录中没有支持的图片: {folder} ({', '.join(sorted(IMAGE_EXTENSIONS))})")
        return

    print(
        f"批量录入: 目录={folder}, pattern={args.pattern}, "
        f"id_prefix={args.id_prefix!r}, suffix_width={args.suffix_width}, dry_run={dry}"
    )

    for image_path in files:
        stem = image_path.stem
        sid = stem_to_student_id(
            stem,
            args.pattern,
            id_prefix=args.id_prefix,
            suffix_width=args.suffix_width,
        )
        if sid is None:
            skip.append((image_path.name, "文件名无法解析为学号"))
            continue

        if dry:
            print(f"  [dry-run] {image_path.name} -> {sid}")
            ok_count += 1
            continue

        ok, msg = register_one_image(sid, image_path)
        if ok:
            stu = Student.query.get(sid)
            print(f"  OK {image_path.name} -> {sid} ({stu.name if stu else ''}) {msg}")
            ok_count += 1
        else:
            errors.append((f"{image_path.name}->{sid}", msg))

    if not dry:
        init_face_encodings()

    print(f"完成: 成功处理 {ok_count} 个文件, 跳过 {len(skip)}, 失败 {len(errors)}")
    for name, reason in skip:
        print(f"  SKIP {name}: {reason}")
    for name, reason in errors:
        print(f"  FAIL {name}: {reason}")


def main() -> None:
    args = parse_args()
    app = create_app()

    with app.app_context():
        if args is None:
            # 兼容旧脚本：常量单张录入
            image_path = Path(IMAGE_PATH)
            print("使用内置常量单张录入（建议改用 --student-id / --image 或 --bulk）")
            ok, msg = register_one_image(STUDENT_ID, image_path)
            if not ok:
                raise RuntimeError(msg)
            stu = Student.query.get(STUDENT_ID)
            print("注册成功:", STUDENT_ID, getattr(stu, "name", ""), msg)
            init_face_encodings()
            return

        if args.bulk is not None:
            run_bulk(args)
            return

        if not args.student_id or not args.image:
            raise SystemExit("单张模式请同时指定 --student-id 与 --image")

        image_path = args.image.resolve()
        if args.dry_run:
            print(f"[dry-run] would register {args.student_id} <- {image_path}")
            return

        ok, msg = register_one_image(args.student_id, image_path)
        if not ok:
            raise RuntimeError(msg)
        stu = Student.query.get(args.student_id)
        print("注册成功:", args.student_id, getattr(stu, "name", ""), msg)
        init_face_encodings()


if __name__ == "__main__":
    main()
