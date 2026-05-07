from ..models.emotion_record import EmotionRecord
from ..database.db import db
from sqlalchemy import func

def get_emotion_statistics():
    """统计各种情绪数量"""
    result = db.session.query(
        EmotionRecord.emotion_type,
        func.count(EmotionRecord.id)
    ).group_by(EmotionRecord.emotion_type).all()
    return {emotion: count for emotion, count in result}

def get_emotion_by_student(student_id):
    return EmotionRecord.query.filter_by(student_id=student_id).all()