from pathlib import Path
from flask import Blueprint, request
from ..utils.auth_decorator import role_required
from ..utils.response import success, error
from ..models.student import Student
from ..database.db import db
import base64
import pickle
import numpy as np

student_bp = Blueprint('student', __name__, url_prefix='/api/students')

FACE_PHOTO_DIR = Path(__file__).resolve().parents[2] / "backend" / "data" / "face_photos"


@student_bp.route('/', methods=['GET'])
@role_required('teacher')
def list_students():
    students = Student.query.all()
    return success([
        {
            'student_id': s.student_id,
            'name': s.name,
            'class_': s.class_,
            'has_face': bool(s.face_encoding),
        }
        for s in students
    ])


@student_bp.route('/<student_id>/face', methods=['DELETE'])
@role_required('teacher')
def delete_face(student_id):
    stu = Student.query.get(student_id)
    if not stu:
        return error(404, f'学号不存在: {student_id}')
    if not stu.face_encoding:
        return error(400, f'该学生尚未录入人脸')

    # 1. 清除数据库中的人脸特征
    stu.face_encoding = None
    db.session.commit()

    # 2. 同步更新内存缓存（核心修复）
    from .. import face_recognition as fr
    fr.face_encodings_db.pop(str(student_id), None)
    fr.face_student_names.pop(str(student_id), None)

    # 3. 删除照片文件（如果存在）
    photo_path = FACE_PHOTO_DIR / f"{student_id}.jpg"
    if photo_path.exists():
        photo_path.unlink()

    return success({'student_id': student_id, 'name': stu.name})


@student_bp.route('/diagnose', methods=['POST'])
@role_required('teacher')
def diagnose_face():
    """上传一帧图片，返回其与库中每位学生的余弦距离，用于判断误识别原因。"""
    from .. import face_recognition as fr

    data = request.get_json()
    image_b64 = data.get('image')
    if not image_b64:
        return error(400, '缺少图片')

    if not fr.face_encodings_db:
        fr.init_face_encodings()

    try:
        image, _ = fr._decode_base64_image(image_b64)
        emb = fr.extract_primary_embedding(image)
    except Exception as exc:
        return error(400, f'人脸提取失败: {exc}')

    rows = []
    for sid, templates in fr.face_encodings_db.items():
        normalized = fr._normalize_encoding_list(templates)
        if not normalized:
            continue
        dists = [fr._cosine_distance(emb, np.asarray(t, dtype=np.float32)) for t in normalized]
        min_dist = min(dists)
        rows.append({
            'student_id': sid,
            'name': fr.face_student_names.get(sid, ''),
            'template_count': len(normalized),
            'min_distance': round(min_dist, 4),
            'all_distances': [round(d, 4) for d in dists],
            'would_match': min_dist <= fr.MATCH_THRESHOLD,
        })

    rows.sort(key=lambda r: r['min_distance'])
    return success({'threshold': fr.MATCH_THRESHOLD, 'results': rows})


@student_bp.route('/<student_id>/face/reset', methods=['POST'])
@role_required('teacher')
def reset_face_templates(student_id):
    """
    清空该学生的所有模板并从保存的注册照片重新提取单条干净嵌入。
    用于修复模板污染（他人 embedding 混入）的问题。
    """
    from .. import face_recognition as fr

    stu = Student.query.get(student_id)
    if not stu:
        return error(404, f'学号不存在: {student_id}')

    photo_path = FACE_PHOTO_DIR / f"{student_id}.jpg"
    if not photo_path.exists():
        return error(400, '找不到该学生的注册照片，无法重建模板。请先重新注册人脸。')

    import cv2
    img = cv2.imread(str(photo_path))
    if img is None:
        return error(400, '注册照片读取失败')

    try:
        emb = fr.extract_primary_embedding(img)
    except Exception as exc:
        return error(400, f'人脸提取失败: {exc}')

    clean_templates = [emb]
    stu.face_encoding = base64.b64encode(pickle.dumps(clean_templates)).decode('utf-8')
    db.session.commit()

    # 同步更新内存缓存
    fr.face_encodings_db[str(student_id)] = clean_templates
    fr.face_student_names[str(student_id)] = str(stu.name)

    return success({
        'student_id': student_id,
        'name': stu.name,
        'template_count': 1,
        'embedding_dim': int(np.asarray(emb).size),
    })
