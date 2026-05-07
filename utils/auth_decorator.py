from functools import wraps
from flask import request, g
from flask_jwt_extended import jwt_required, get_jwt_identity
from ..models.user import User
from ..utils.response import error

def role_required(required_role):
    def decorator(fn):
        @wraps(fn)
        @jwt_required()
        def wrapper(*args, **kwargs):
            current_user_id = int(get_jwt_identity())
            user = User.query.get(current_user_id)
            if not user or user.role != required_role:
                return error(403, '权限不足')
            g.current_user = user
            return fn(*args, **kwargs)
        return wrapper
    return decorator

def login_required(fn):
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        current_user_id = int(get_jwt_identity())
        user = User.query.get(current_user_id)
        if not user:
            return error(401, '用户不存在')
        g.current_user = user
        return fn(*args, **kwargs)
    return wrapper