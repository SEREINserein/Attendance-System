import datetime
from ..database.db import db
from ..models.attendance_record import AttendanceRecord
from ..models.emotion_record import EmotionRecord
from ..face_recognition import compare_faces, detect_liveness_photo
from ..liveness import detect_liveness_video
from ..emotion import analyze_emotion
from ..utils.exceptions import LivenessCheckFailed, FaceRecognitionTimeout

def process_attendance_photo(image_base64: str, is_video_mode=False, frames=None):
    """
    处理单张照片考勤或视频流考勤
    返回: {
        'success': bool,
        'student_id': str,
        'name': str,
        'emotion': str,
        'liveness_pass': bool,
        'message': str
    }
    """
    # 1. 活体检测（照片攻击）
    photo_liveness = detect_liveness_photo(image_base64)
    if not photo_liveness['is_live']:
        raise LivenessCheckFailed(400, f'照片攻击检测未通过: {photo_liveness["reason"]}')
    
    # 2. 如果是视频模式，进行视频活体检测（D同学实现）
    if is_video_mode and frames:
        video_liveness = detect_liveness_video(frames)
        if not video_liveness['is_live']:
            raise LivenessCheckFailed(400, f'视频活体检测未通过: {video_liveness["attack_type"]}')
    
    # 3. 人脸比对
    compare_result = compare_faces(image_base64)
    if not compare_result['success']:
        return {
            'success': False,
            'student_id': None,
            'name': None,
            'emotion': None,
            'liveness_pass': True,
            'message': compare_result['error_msg']
        }
    
    student = compare_result['matched_student']
    # 4. 情绪分析
    emotion_result = analyze_emotion(image_base64)
    emotion = emotion_result['emotion']
    
    # 5. 保存考勤记录
    now = datetime.datetime.now()
    record = AttendanceRecord(
        student_id=student['student_id'],
        check_time=now,
        status='success',
        liveness_result='photo_live_pass',
        emotion=emotion
    )
    db.session.add(record)
    # 保存情绪记录
    emotion_record = EmotionRecord(
        student_id=student['student_id'],
        record_time=now,
        emotion_type=emotion,
        source='attendance'
    )
    db.session.add(emotion_record)
    db.session.commit()
    
    return {
        'success': True,
        'student_id': student['student_id'],
        'name': student['name'],
        'emotion': emotion,
        'liveness_pass': True,
        'message': '考勤成功'
    }