from ..database.db import db

class AttendanceRecord(db.Model):
    __tablename__ = 'attendance_records'
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), nullable=False)
    check_time = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(10), nullable=False)
    liveness_result = db.Column(db.String(100))
    emotion = db.Column(db.String(20))