from ..database.db import db

class ActivityParticipation(db.Model):
    __tablename__ = 'activity_participation'
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), nullable=False)
    photo_id = db.Column(db.String(50), nullable=False)
    participate_time = db.Column(db.DateTime, nullable=False)