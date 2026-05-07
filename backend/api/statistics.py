from flask import Blueprint, g
from ..utils.auth_decorator import login_required, role_required
from ..utils.response import success
from ..models.activity_participation import ActivityParticipation
from ..models.student import Student
from ..database.db import db
from sqlalchemy import func

stats_bp = Blueprint('statistics', __name__, url_prefix='/api/statistics')

@stats_bp.route('/activity', methods=['GET'])
@role_required('teacher')
def activity_freq():
    # 统计每个学生参与合照的次数
    result = db.session.query(
        ActivityParticipation.student_id,
        func.count(ActivityParticipation.id).label('count')
    ).group_by(ActivityParticipation.student_id).all()
    # 补充学生姓名
    data = []
    for sid, cnt in result:
        student = Student.query.get(sid)
        data.append({
            'student_id': sid,
            'name': student.name if student else '未知',
            'participation_count': cnt
        })
    return success(data)