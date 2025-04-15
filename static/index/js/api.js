/**
 * API模块 - 处理与后端的API交互
 */

const APIModule = {
    /**
     * 发送检测请求
     * @param {File} file - 上传的图片文件
     * @param {Array} selectedPieces - 选择的棋子类型
     * @returns {Promise} 检测结果
     */
    detectPieces: async function(file, selectedPieces) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('pieces', JSON.stringify(selectedPieces));

        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('检测失败');
        return await response.json();
    },

    /**
     * 生成棋谱
     * @param {string} sessionId - 会话ID
     * @param {Array} corners - 角点坐标
     * @returns {Promise} 棋谱图片
     */
    generateNotation: async function(sessionId, corners) {
        const formData = new FormData();
        formData.append('session_id', sessionId);
        formData.append('corners', JSON.stringify(corners));

        const response = await fetch('/generate_notation', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('生成棋谱失败');
        }

        return await response.blob();
    },

    /**
     * 发送聊天请求
     * @param {string} prompt - 问题内容
     * @returns {Promise} 聊天响应
     */
    sendChatRequest: async function(prompt) {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: prompt,
                model: 'ollama',
                stream: false
            })
        });

        if (!response.ok) {
            throw new Error('请求失败: ' + response.statusText);
        }
        
        return await response.json();
    }
};

// 导出API模块
// window.APIModule = APIModule;
Object.defineProperty(window, 'APIModule', {
  value: APIModule,
  writable: true,
  configurable: true
});