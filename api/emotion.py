from flask import Blueprint, g
from ..utils.auth_decorator import login_required
from ..utils.response import success
from ..services.emotion_service import get_emotion_statistics

emotion_bp = Blueprint('emotion', __name__, url_prefix='/api/emotion')

@emotion_bp.route('/statistics', methods=['GET'])
@login_required
def statistics():
    stats = get_emotion_statistics()
    return success(stats)