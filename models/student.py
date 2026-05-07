from ..database.db import db

class Student(db.Model):
    __tablename__ = 'students'
    student_id = db.Column(db.String(20), primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    class_ = db.Column('class', db.String(20))
    face_encoding = db.Column(db.Text)  # base64 or pickle

    def to_dict(self):
        return {
            'student_id': self.student_id,
            'name': self.name,
            'class': self.class_
        }