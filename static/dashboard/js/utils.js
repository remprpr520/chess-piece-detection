/**
 * 工具函数文件 - 包含通用功能函数
 */

/**
 * 将Markdown转换为HTML
 * @param {string} markdown - Markdown格式文本
 * @returns {string} - 转换后的HTML
 */
function convertMarkdownToHTML(markdown) {
    if (!markdown) return '';
    return marked.parse(markdown);
}

/**
 * 格式化日期为YYYY-MM-DD字符串
 * @param {Date} date - 日期对象
 * @returns {string} - 格式化后的日期字符串
 */
function formatDate(date) {
    return date.toISOString().split('T')[0];
}

/**
 * 初始化日期选择器为默认值（今天和一周前）
 */
function initDatePickers() {
    // 获取今天和一周前的日期
    const today = new Date();
    const oneWeekAgo = new Date();
    oneWeekAgo.setDate(today.getDate() - 7);

    // 设置日期选择器的默认值
    document.getElementById('detection-end-date').value = formatDate(today);
    document.getElementById('detection-start-date').value = formatDate(oneWeekAgo);
    document.getElementById('qa-end-date').value = formatDate(today);
    document.getElementById('qa-start-date').value = formatDate(oneWeekAgo);
}

/**
 * 获取最近7天的日期数组
 * @returns {Array} - 日期字符串数组
 */
function getLast7Days() {
    const dates = [];
    const today = new Date();
    for (let i = 6; i >= 0; i--) {
        const date = new Date();
        date.setDate(today.getDate() - i);
        dates.push(formatDate(date));
    }
    return dates;
}