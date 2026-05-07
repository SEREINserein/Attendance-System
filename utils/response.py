def success(data=None, message='success'):
    return {
        'code': 200,
        'message': message,
        'data': data
    }

def error(code, message):
    return {
        'code': code,
        'message': message,
        'data': None
    }