from flask import Blueprint, request, g
from ..utils.auth_decorator import login_required
from ..utils.response import success, error
from ..services.attendance_service import process_attendance_photo
from ..models.attendance_record import AttendanceRecord
from ..models.user import User
from ..database.db import db
from datetime import datetime

attendance_bp = Blueprint('attendance', __name__, url_prefix='/api/attendance')

@attendance_bp.route('/check', methods=['POST'])
@login_required
def check_attendance():
    data = request.get_json()
    image_base64 = data.get('image')
    if not image_base64:
        return error(400, '缺少图片数据')
    try:
        result = process_attendance_photo(image_base64, is_video_mode=False, frames=None)
        return success(result)
    except Exception as e:
        return error(500, str(e))

@attendance_bp.route('/records', methods=['GET'])
@login_required
def get_records():
    user = g.current_user
    student_id = request.args.get('student_id')
    date_str = request.args.get('date')
    query = AttendanceRecord.query
    if user.role == 'student':
        # 学生只能看自己的
        query = query.filter_by(student_id=user.student_id)
    if student_id and user.role == 'teacher':
        query = query.filter_by(student_id=student_id)
    if date_str:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        start = datetime(date_obj.year, date_obj.month, date_obj.day)
        end = datetime(date_obj.year, date_obj.month, date_obj.day, 23,59,59)
        query = query.filter(AttendanceRecord.check_time.between(start, end))
    records = query.order_by(AttendanceRecord.check_time.desc()).all()
    return success([{
        'student_id': r.student_id,
        'check_time': r.check_time.strftime('%Y-%m-%d %H:%M:%S'),
        'status': r.status,
        'emotion': r.emotion
    } for r in records])

@attendance_bp.route('/export', methods=['GET'])
@login_required
def export_records():
    user = g.current_user
    if user.role != 'teacher':
        return error(403, '仅教师可导出')
    import pandas as pd
    records = AttendanceRecord.query.all()
    data = [{
        '学号': r.student_id,
        '考勤时间': r.check_time,
        '状态': r.status,
        '情绪': r.emotion
    } for r in records]
    df = pd.DataFrame(data)
    # 保存为临时excel文件
    filepath = 'temp_attendance.xlsx'
    df.to_excel(filepath, index=False)
    from flask import send_file
    return send_file(filepath, as_attachment=True, download_name='考勤记录.xlsx')