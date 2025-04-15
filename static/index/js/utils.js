/**
 * 工具函数模块 - 提供各种通用功能
 */

const ChessUtils = {
    /**
     * 将Markdown文本转换为HTML
     * @param {string} markdown - 要转换的Markdown文本
     * @returns {string} 转换后的HTML
     */
    convertMarkdownToHTML: function(markdown) {
        if (!markdown) return '';
        return marked.parse(markdown);
    },

    /**
     * 获取当前时间的格式化字符串
     * @returns {string} 格式化的时间字符串 (HH:MM)
     */
    getCurrentTime: function() {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    },

    /**
     * 显示DOM元素
     * @param {string} elementId - 元素ID
     */
    showElement: function(elementId) {
        document.getElementById(elementId).classList.remove('hidden');
    },

    /**
     * 隐藏DOM元素
     * @param {string} elementId - 元素ID
     */
    hideElement: function(elementId) {
        document.getElementById(elementId).classList.add('hidden');
    },

    /**
     * 隐藏多个DOM元素
     * @param {Array<string>} elementIds - 元素ID数组
     */
    hideElements: function(elementIds) {
        elementIds.forEach(id => this.hideElement(id));
    }
};

// 导出工具函数，使其可以在其他模块中使用
//  window.ChessUtils = ChessUtils;
Object.defineProperty(window, 'ChessUtils', {
  value: ChessUtils,
  writable: true,
  configurable: true
});