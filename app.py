from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from .database.db import db
from .utils.exceptions import BusinessException
from .utils.response import error

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['JWT_SECRET_KEY'] = 'your-secret-key-change-in-production'
    CORS(app)
    JWTManager(app)
    db.init_app(app)
    
    # 注册蓝图
    from .api.auth import auth_bp
    from .api.attendance import attendance_bp
    from .api.photo import photo_bp
    from .api.emotion import emotion_bp
    from .api.statistics import stats_bp
    from .api.student import student_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(attendance_bp)
    app.register_blueprint(photo_bp)
    app.register_blueprint(emotion_bp)
    app.register_blueprint(stats_bp)
    app.register_blueprint(student_bp)
    
    # 全局异常处理
    @app.errorhandler(BusinessException)
    def handle_business_exception(e):
        return jsonify(error(e.code, e.message)), e.code
    
    @app.errorhandler(Exception)
    def handle_general_exception(e):
        app.logger.error(f'Unexpected error: {e}')
        return jsonify(error(500, '服务器内部错误')), 500
    
    # 创建数据库表（若不存在）
    with app.app_context():
        db.create_all()
        # 初始化测试数据（教师、学生、人脸库等）
        init_test_data()
        # 加载人脸库
        from .face_recognition import init_face_encodings
        init_face_encodings()
    
    return app

def init_test_data():
    from .models.user import User
    from .models.student import Student
    from werkzeug.security import generate_password_hash
    import base64
    import pickle
    # 检查是否已存在
    if User.query.first():
        return
    # 创建教师
    teacher = User(username='teacher', password_hash=generate_password_hash('123456'), role='teacher')
    # 创建学生
    students_data = [
        ('2024001', '张三', '计科1班'),
        ('2024002', '李四', '计科1班'),
        ('2024003', '王五', '计科1班'),
    ]
    db.session.add(teacher)
    for sid, name, cls in students_data:
        stu = Student(student_id=sid, name=name, class_=cls)
        db.session.add(stu)
        user = User(username=sid, password_hash=generate_password_hash('123456'), role='student', student_id=sid)
        db.session.add(user)
    db.session.commit()
    # 人脸特征留空，由 register_face_once.py 用真实照片录入