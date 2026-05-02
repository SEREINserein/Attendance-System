window.onload = function() {
    // 初始化 Flatpickr 日历
    flatpickr("#searchDate", {
        dateFormat: "Y-m-d",
        locale: "zh",
        disableMobile: true,
    });

    // ---------- DOM 元素 ----------
    const video = document.getElementById('video');
    const captureBtn = document.getElementById('captureBtn');
    const autoCheckbox = document.getElementById('autoModeCheckbox');
    const uploadGroupBtn = document.getElementById('uploadGroupBtn');
    const searchRecordsBtn = document.getElementById('searchRecordsBtn');
    const exportExcelBtn = document.getElementById('exportExcelBtn');
    const refreshActivityChartBtn = document.getElementById('refreshActivityChart');
    const chartTypeSelect = document.getElementById('chartType');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const roleBtns = document.querySelectorAll('.role-btn');

    // ---------- 全局变量 ----------
    let currentRole = 'teacher';
    let stream = null;
    let emotionChart = null;
    let activityChart = null;
    let autoModeInterval = null;
    let lastAutoAttendanceTime = 0;
    let isAutoMode = false;

    // ---------- Mock 考勤数据（增加更多日期记录以便展示时间趋势）----------
    let attendanceRecordsMock = [
        { name: "张三", student_id: "2024001", time: "2026-05-01 09:00", status: "成功", emotion: "happy", participate_count: 5 },
        { name: "李四", student_id: "2024002", time: "2026-05-01 09:02", status: "成功", emotion: "neutral", participate_count: 3 },
        { name: "王五", student_id: "2024003", time: "2026-05-02 08:55", status: "成功", emotion: "sad", participate_count: 7 },
        { name: "张三", student_id: "2024001", time: "2026-05-02 09:10", status: "成功", emotion: "happy", participate_count: 5 },
        { name: "李四", student_id: "2024002", time: "2026-05-02 09:15", status: "成功", emotion: "neutral", participate_count: 3 },
        { name: "李四", student_id: "2024002", time: "2026-05-03 14:20", status: "成功", emotion: "happy", participate_count: 3 },
        { name: "李四", student_id: "2024002", time: "2026-05-05 10:30", status: "成功", emotion: "sad", participate_count: 3 }
    ];

    // 教师视图：获取每个学生的总参与次数
    function getStudentTotalCounts() {
        let map = new Map();
        attendanceRecordsMock.forEach(rec => {
            if (!map.has(rec.student_id)) {
                map.set(rec.student_id, { name: rec.name, count: rec.participate_count });
            }
        });
        return map;
    }

    // 学生视图：获取指定学生每日活动次数（按考勤日期统计）
    function getStudentDailyCounts(studentId) {
        const records = attendanceRecordsMock.filter(r => r.student_id === studentId);
        const dailyMap = new Map();
        records.forEach(rec => {
            const date = rec.time.split(' ')[0];
            dailyMap.set(date, (dailyMap.get(date) || 0) + 1);
        });
        const sortedDates = Array.from(dailyMap.keys()).sort();
        const labels = sortedDates;
        const data = sortedDates.map(date => dailyMap.get(date));
        return { labels, data };
    }

    // 核心：绘制活动图表（教师视图=总次数学生对比，学生视图=个人每日趋势）
    function renderActivityChart(chartType) {
        const canvas = document.getElementById('activityChart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (activityChart) {
            activityChart.destroy();
            activityChart = null;
        }

        let chartLabels = [];
        let chartData = [];
        let chartLabelText = '';

        if (currentRole === 'teacher') {
            const totalMap = getStudentTotalCounts();
            for (let [id, info] of totalMap.entries()) {
                chartLabels.push(`${info.name}(${id})`);
                chartData.push(info.count);
            }
            chartLabelText = '总参与次数';
        } else {
            const studentId = "2024002"; // 李四（学生视图固定为当前登录学生）
            const { labels, data } = getStudentDailyCounts(studentId);
            chartLabels = labels;
            chartData = data;
            chartLabelText = '当日参与次数';
        }

        activityChart = new Chart(ctx, {
            type: chartType,
            data: {
                labels: chartLabels,
                datasets: [{
                    label: chartLabelText,
                    data: chartData,
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        stepSize: 1,
                        title: { display: true, text: currentRole === 'teacher' ? '参与次数' : '参与次数 (次/天)' }
                    },
                    x: {
                        title: { display: true, text: currentRole === 'teacher' ? '学生' : '日期' }
                    }
                }
            }
        });
    }

    // 实时切换图表类型
    if (chartTypeSelect) {
        chartTypeSelect.addEventListener('change', function() {
            const activeTab = document.querySelector('.tab-btn.active')?.getAttribute('data-tab');
            if (activeTab === 'records') {
                renderActivityChart(chartTypeSelect.value);
            }
        });
    }
    if (refreshActivityChartBtn) {
        refreshActivityChartBtn.addEventListener('click', function() {
            renderActivityChart(chartTypeSelect.value);
        });
    }

    // ---------- 摄像头初始化 ----------
    async function initCamera() {
        if (!video) return;
        try {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            console.log("摄像头已启动");
        } catch (err) {
            console.error("摄像头调用失败:", err);
            alert("无法调用摄像头，请检查权限或摄像头设备");
        }
    }
    initCamera();

    // 获取当前角色对应的学生信息（Mock 考勤识别用）
    function getCurrentStudentInfo() {
        if (currentRole === 'teacher') {
            return { name: "张三", student_id: "2024001", emotion: "happy" };
        } else {
            return { name: "李四", student_id: "2024002", emotion: "neutral" };
        }
    }

    // 截取视频帧
    function captureVideoFrame() {
        if (!video || video.videoWidth === 0) return null;
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/jpeg');
    }

    // 通用考勤逻辑（手动/自动）
    async function performAttendance(imageBase64, isAuto = false) {
        const resultDiv = document.getElementById('attendanceResult');
        if (!imageBase64) {
            resultDiv.innerHTML = "未获取到图像，请重试";
            return false;
        }
        return new Promise((resolve) => {
            setTimeout(() => {
                const student = getCurrentStudentInfo();
                const now = Date.now();
                if (lastAutoAttendanceTime && (now - lastAutoAttendanceTime) < 5000) {
                    resultDiv.innerHTML = `⏰ ${isAuto ? '自动检测' : '手动'} : 5秒内已打卡过，请勿重复。`;
                    resolve(false);
                    return;
                }
                const mockResult = {
                    success: true,
                    student_name: student.name,
                    student_id: student.student_id,
                    time: new Date().toLocaleString(),
                    liveness: true,
                    emotion: student.emotion
                };
                if (mockResult.success) {
                    resultDiv.innerHTML = `
                        <strong>${isAuto ? '🤖 自动打卡成功！' : '✅ 考勤成功！'}</strong><br>
                        姓名：${mockResult.student_name}<br>
                        学号：${mockResult.student_id}<br>
                        时间：${mockResult.time}<br>
                        活体检测：${mockResult.liveness ? "通过" : "失败"}<br>
                        情绪：${mockResult.emotion}
                    `;
                    lastAutoAttendanceTime = now;
                    const activeTab = document.querySelector('.tab-btn.active')?.getAttribute('data-tab');
                    if (activeTab === 'records') {
                        document.getElementById('searchRecordsBtn').click();
                    }
                    resolve(true);
                } else {
                    resultDiv.innerHTML = `${isAuto ? '自动' : '手动'}考勤失败，请重试`;
                    resolve(false);
                }
            }, 500);
        });
    }

    // 手动拍照
    if (captureBtn) {
        captureBtn.addEventListener('click', async () => {
            const img = captureVideoFrame();
            if (!img) {
                document.getElementById('attendanceResult').innerHTML = "摄像头未就绪";
                return;
            }
            await performAttendance(img, false);
        });
    }

    // 自动模式开关
    if (autoCheckbox) {
        autoCheckbox.addEventListener('change', (e) => {
            isAutoMode = e.target.checked;
            if (isAutoMode) {
                if (autoModeInterval) clearInterval(autoModeInterval);
                autoModeInterval = setInterval(async () => {
                    if (!isAutoMode) return;
                    const img = captureVideoFrame();
                    if (img) await performAttendance(img, true);
                }, 2000);
            } else {
                if (autoModeInterval) {
                    clearInterval(autoModeInterval);
                    autoModeInterval = null;
                }
            }
        });
    }

    // 合照上传识别
    if (uploadGroupBtn) {
        uploadGroupBtn.addEventListener('click', async () => {
            const fileInput = document.getElementById('groupPhoto');
            const file = fileInput.files[0];
            if (!file) {
                alert("请先选择一张合照图片");
                return;
            }
            const reader = new FileReader();
            reader.onloadend = function() {
                document.getElementById('groupResult').innerHTML = "正在识别合影中的学生...";
                setTimeout(() => {
                    const allStudents = [
                        { name: "张三", student_id: "2024001" },
                        { name: "李四", student_id: "2024002" },
                        { name: "王五", student_id: "2024003" }
                    ];
                    let list = [];
                    if (currentRole === 'teacher') {
                        list = allStudents;
                    } else {
                        list = allStudents.filter(s => s.student_id === "2024002");
                    }
                    let html = "<h4>识别出的学生名单：</h4><ul>";
                    list.forEach(s => { html += `<li>${s.name} (${s.student_id})</li>`; });
                    html += "</ul>";
                    document.getElementById('groupResult').innerHTML = html;
                }, 800);
            };
            reader.readAsDataURL(file);
        });
    }

    // 渲染考勤表格（含活动参与频次列，仍展示原有 participate_count 值）
    function renderRecordsTable(records) {
        const container = document.getElementById('recordsList');
        if (!records.length) {
            container.innerHTML = "<p>暂无考勤记录</p>";
            return;
        }
        let html = '<table><thead>\n<th>姓名</th><th>学号</th><th>考勤时间</th><th>状态</th><th>情绪</th><th>活动参与频次</th>\n</thead><tbody>';
        records.forEach(r => {
            html += `<tr>
                        <td>${r.name}</td>
                        <td>${r.student_id}</td>
                        <td>${r.time}</td>
                        <td>${r.status}</td>
                        <td>${r.emotion}</td>
                        <td>${r.participate_count}</td>
                     </tr>`;
        });
        html += '</tbody></table>';
        container.innerHTML = html;
    }

    // 查询考勤记录
    if (searchRecordsBtn) {
        searchRecordsBtn.addEventListener('click', () => {
            const studentId = document.getElementById('searchStudentId').value;
            const date = document.getElementById('searchDate').value;
            let filtered = [...attendanceRecordsMock];
            if (currentRole === 'student') {
                filtered = filtered.filter(r => r.student_id === "2024002");
            }
            if (studentId) {
                filtered = filtered.filter(r => r.student_id.includes(studentId));
            }
            if (date) {
                filtered = filtered.filter(r => r.time.startsWith(date));
            }
            renderRecordsTable(filtered);
        });
    }

    // 导出 Excel
    if (exportExcelBtn) {
        exportExcelBtn.addEventListener('click', () => {
            const table = document.querySelector('#recordsList table');
            if (!table) { alert("暂无数据可导出"); return; }
            let csv = [];
            const headers = Array.from(table.querySelectorAll('thead th')).map(th => th.innerText);
            csv.push(headers.join(','));
            const rows = table.querySelectorAll('tbody tr');
            for (let row of rows) {
                const cols = row.querySelectorAll('td');
                const rowData = Array.from(cols).map(cell => cell.innerText);
                csv.push(rowData.join(','));
            }
            const blob = new Blob(["\uFEFF" + csv.join('\n')], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = '考勤记录_活动统计.csv';
            link.click();
        });
    }

    // 情绪图表
    async function loadEmotionStats() {
        const stats = currentRole === 'teacher'
            ? { happy: 15, sad: 3, neutral: 8, angry: 1, surprise: 4 }
            : { happy: 4, neutral: 1, sad: 0 };
        const ctx = document.getElementById('emotionChart').getContext('2d');
        if (emotionChart) emotionChart.destroy();
        emotionChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(stats),
                datasets: [{ label: '情绪出现次数', data: Object.values(stats), backgroundColor: 'rgba(54, 162, 235, 0.5)' }]
            },
            options: { responsive: true, scales: { y: { beginAtZero: true, stepSize: 1 } } }
        });
    }

    // 选项卡切换
    function switchTab(tabId) {
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        const activeContent = document.getElementById(tabId);
        if (activeContent) activeContent.classList.add('active');
        tabBtns.forEach(btn => btn.classList.remove('active'));
        const activeBtn = Array.from(tabBtns).find(btn => btn.getAttribute('data-tab') === tabId);
        if (activeBtn) activeBtn.classList.add('active');

        if (tabId === 'emotion') loadEmotionStats();
        if (tabId === 'records') {
            if (searchRecordsBtn) searchRecordsBtn.click();
            if (chartTypeSelect) renderActivityChart(chartTypeSelect.value);
        }
    }

    tabBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const tabId = btn.getAttribute('data-tab');
            switchTab(tabId);
        });
    });

    // 角色切换
    function setRole(role) {
        currentRole = role;
        roleBtns.forEach(btn => btn.classList.remove('active'));
        const activeRoleBtn = Array.from(roleBtns).find(btn => btn.getAttribute('data-role') === role);
        if (activeRoleBtn) activeRoleBtn.classList.add('active');

        const activeTabId = document.querySelector('.tab-btn.active')?.getAttribute('data-tab');
        if (activeTabId === 'records') {
            if (searchRecordsBtn) searchRecordsBtn.click();
            if (chartTypeSelect) renderActivityChart(chartTypeSelect.value);
        } else if (activeTabId === 'emotion') {
            loadEmotionStats();
        } else if (activeTabId === 'attendance') {
            document.getElementById('attendanceResult').innerHTML = '角色已切换，拍照考勤将使用新角色权限';
            lastAutoAttendanceTime = 0;
        } else if (activeTabId === 'group') {
            document.getElementById('groupResult').innerHTML = '角色已切换，合照识别结果将按新角色显示';
        }
    }

    roleBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const role = btn.getAttribute('data-role');
            setRole(role);
        });
    });

    // 默认显示考勤标签页
    switchTab('attendance');
};