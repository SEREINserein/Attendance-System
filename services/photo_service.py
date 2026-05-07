import datetime
import uuid
from ..face_recognition import recognize_group_photo
from ..emotion import analyze_emotion
from ..database.db import db
from ..models.activity_participation import ActivityParticipation
from ..models.emotion_record import EmotionRecord

def process_group_photo(image_base64: str):
    print("=" * 50)
    try:
        from .. import face_recognition as fr
        print("DEBUG: face_encodings_db 学生数 =", len(fr.face_encodings_db))
        print("DEBUG: 学生ID列表 =", list(fr.face_encodings_db.keys()))
        print("DEBUG: DeepFace =", fr.DeepFace)
        
        # 手动触发一次初始化
        if not fr.face_encodings_db:
            print("DEBUG: 人脸库为空，手动调用 init_face_encodings()")
            fr.init_face_encodings()
            print("DEBUG: 初始化后学生数 =", len(fr.face_encodings_db))
            
    except Exception as e:
        print("DEBUG: 检查时出错 =", repr(e))
    print("=" * 50)

    students = recognize_group_photo(image_base64)
    print("DEBUG: recognize_group_photo 返回 =", students)

    photo_id = str(uuid.uuid4())
    now = datetime.datetime.now()
    for stu in students:
        part = ActivityParticipation(
            student_id=stu['student_id'],
            photo_id=photo_id,
            participate_time=now
        )
        db.session.add(part)
    db.session.commit()
    return students