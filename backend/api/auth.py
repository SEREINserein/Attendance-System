from flask import Blueprint, request, g
from flask_jwt_extended import create_access_token
from werkzeug.security import generate_password_hash, check_password_hash
from ..models.user import User
from ..database.db import db
from ..utils.response import success, error

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return error(400, '用户名密码不能为空')
    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password_hash, password):
        return error(401, '用户名或密码错误')
    access_token = create_access_token(identity=str(user.id))
    return success({
        'token': access_token,
        'role': user.role,
        'student_id': user.student_id
    })