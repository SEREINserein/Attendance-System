class BusinessException(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message

class FaceRecognitionTimeout(BusinessException):
    pass

class UploadFailedError(BusinessException):
    pass

class LivenessCheckFailed(BusinessException):
    pass