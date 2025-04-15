/**
 * 图表相关函数
 */

/**
 * 创建检测趋势图表
 * @param {Object} data - 包含统计数据的对象
 */
function createDetectionChart(data) {
    // 创建图表上下文
    const ctx = document.getElementById('detection-chart').getContext('2d');

    // 获取最近7天的日期
    const dates = getLast7Days();

    // 从后端数据中获取每日统计
    const detections = dates.map(date => {
        const dailyStats = data?.daily_stats?.[date] || { detections: 0, notations: 0 };
        return dailyStats.detections + dailyStats.notations;
    });

    const questions = dates.map(date => {
        const dailyStats = data?.daily_stats?.[date] || { questions: 0 };
        return dailyStats.questions;
    });

    // 中文日期格式化
    const formattedDates = dates.map(date => moment(date).format('MM月DD日'));

    // 创建图表
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: formattedDates,
            datasets: [
                {
                    label: '检测次数',
                    data: detections,
                    borderColor: '#4F46E5',
                    backgroundColor: 'rgba(79, 70, 229, 0.1)',
                    tension: 0.3,
                    fill: true
                },
                {
                    label: '问答次数',
                    data: questions,
                    borderColor: '#10B981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.3,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}