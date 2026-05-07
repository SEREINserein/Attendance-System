from flask import Blueprint, request, g
from ..utils.auth_decorator import role_required
from ..utils.response import success, error
from ..services.photo_service import process_group_photo

photo_bp = Blueprint('photo', __name__, url_prefix='/api/photo')

@photo_bp.route('/upload', methods=['POST'])
@role_required('teacher')
def upload_group_photo():
    data = request.get_json()
    image_base64 = data.get('image')
    if not image_base64:
        return error(400, '缺少图片')

    print("=" * 50) 
    students = process_group_photo(image_base64)
    print("DEBUG 识别结果:", students)                  
    print("=" * 50)

    return success({'students': students})