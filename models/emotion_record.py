from ..database.db import db

class EmotionRecord(db.Model):
    __tablename__ = 'emotion_records'
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), nullable=False)
    record_time = db.Column(db.DateTime, nullable=False)
    emotion_type = db.Column(db.String(20), nullable=False)
    source = db.Column(db.String(10), nullable=False)