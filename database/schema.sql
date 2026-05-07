-- 用户表（教师/学生统一）
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(10) CHECK (role IN ('teacher', 'student')) NOT NULL,
    student_id VARCHAR(20) NULL
);

-- 学生详细信息
CREATE TABLE students (
    student_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    class VARCHAR(20),
    face_encoding TEXT   -- 存储人脸特征向量的base64或pickle串
);

-- 考勤记录
CREATE TABLE attendance_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id VARCHAR(20) NOT NULL,
    check_time DATETIME NOT NULL,
    status VARCHAR(10) NOT NULL,
    liveness_result TEXT,
    emotion VARCHAR(20),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

-- 情绪记录（用于统计）
CREATE TABLE emotion_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id VARCHAR(20) NOT NULL,
    record_time DATETIME NOT NULL,
    emotion_type VARCHAR(20) NOT NULL,
    source VARCHAR(10) CHECK (source IN ('attendance', 'group_photo')),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

-- 合照参与记录（活动频次）
CREATE TABLE activity_participation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id VARCHAR(20) NOT NULL,
    photo_id VARCHAR(50) NOT NULL,
    participate_time DATETIME NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);