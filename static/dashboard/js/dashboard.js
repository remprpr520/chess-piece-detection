/**
 * 主要的控制逻辑
 */

/**
 * 加载概要统计数据
 */
async function loadSummaryStats() {
    try {
        const response = await fetch('/api/stats/summary');
        const data = await response.json();

        // 更新统计卡片的数据
        document.getElementById('total-detections').textContent = data.total_detections + data.total_notations;
        document.getElementById('today-detections').textContent = data.today_detections + data.today_notations;
        document.getElementById('total-questions').textContent = data.total_questions;
        document.getElementById('today-questions').textContent = data.today_questions;

        // 创建趋势图表
        createDetectionChart(data);

        return data;
    } catch (error) {
        console.error('加载统计数据失败:', error);
        return null;
    }
}

/**
 * 设置选项卡切换
 */
function setupTabs() {
    const tabDetection = document.getElementById('tab-detection');
    const tabQA = document.getElementById('tab-qa');
    const detectionHistory = document.getElementById('detection-history');
    const qaHistory = document.getElementById('qa-history');

    // 检测历史选项卡点击事件
    tabDetection.addEventListener('click', () => {
        tabDetection.classList.add('tab-active');
        tabQA.classList.remove('tab-active');
        detectionHistory.classList.remove('hidden');
        qaHistory.classList.add('hidden');
        loadDetectionHistory(); // 重新加载检测历史
    });

    // 问答历史选项卡点击事件
    tabQA.addEventListener('click', () => {
        tabQA.classList.add('tab-active');
        tabDetection.classList.remove('tab-active');
        qaHistory.classList.remove('hidden');
        detectionHistory.classList.add('hidden');
        loadQAHistory(); // 加载问答历史
    });
}

/**
 * 页面初始化
 */
document.addEventListener('DOMContentLoaded', async () => {
    // 初始化日期选择器
    initDatePickers();

    // 设置选项卡切换
    setupTabs();

    // 加载概要统计数据
    const summaryData = await loadSummaryStats();

    // 加载检测历史数据
    loadDetectionHistory();

    // 绑定搜索按钮事件
    document.getElementById('detection-search').addEventListener('click', loadDetectionHistory);
    document.getElementById('qa-search').addEventListener('click', loadQAHistory);

    // 设置自动刷新（每5分钟刷新一次数据）
    setInterval(loadSummaryStats, 300000);
});