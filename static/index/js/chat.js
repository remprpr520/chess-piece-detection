/**
 * 聊天模块 - 处理聊天界面和交互功能
 */

const ChatModule = {
    // 防止重复提交标志
    isProcessingQuestion: false,

    /**
     * 设置聊天相关的事件
     */
    setupChatEvents: function() {
        // 防止重复绑定事件
        const sendButton = document.getElementById('send-question');
        if (sendButton && !sendButton.hasEventListener) {
            sendButton.addEventListener('click', () => this.sendQuestion());
            sendButton.hasEventListener = true;
        }

        // 问题输入框回车事件
        const questionInput = document.getElementById('question-input');
        if (questionInput && !questionInput.hasEventListener) {
            questionInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.sendQuestion();
                }
            });
            questionInput.hasEventListener = true;
        }

        // 快速问题按钮点击事件
        document.querySelectorAll('.quick-question').forEach(button => {
            if (!button.hasEventListener) {
                button.addEventListener('click', function() {
                    document.getElementById('question-input').value = this.textContent;
                    ChatModule.sendQuestion();
                });
                button.hasEventListener = true;
            }
        });
    },

    /**
     * 发送问题到后端
     */
    sendQuestion: function() {
        // 防止重复提交
        if (this.isProcessingQuestion) return;

        const questionInput = document.getElementById('question-input');
        const question = questionInput.value.trim();

        if (question === '') return;

        // 设置正在处理标志
        this.isProcessingQuestion = true;

        // 显示用户问题
        this.addUserMessage(question);

        // 清空输入框
        questionInput.value = '';

        // 显示AI"思考中"的消息
        const thinkingId = this.addThinkingMessage();

        // 添加关于国际象棋的上下文增强问题
        const enhancedPrompt = `你是一个国际象棋专家助手。请回答以下关于国际象棋的问题：${question}`;

        // 发送请求到后端
        APIModule.sendChatRequest(enhancedPrompt)
            .then(data => {
                // 移除思考中的消息
                document.getElementById(thinkingId).remove();

                // 显示AI回答
                this.addAIMessage(data.answer_content, data.think_content);

                // 为AI回答添加播放按钮
                this.addPlayButtonToLastMessage(data.answer_content);

                // 重置处理标志
                this.isProcessingQuestion = false;
            })
            .catch(error => {
                console.error('发送问题失败:', error);

                // 移除思考中的消息
                document.getElementById(thinkingId).remove();

                // 显示错误消息
                this.addErrorMessage('发送问题失败: ' + error.message);

                // 重置处理标志
                this.isProcessingQuestion = false;
            });
    },

    /**
     * 为最后一条AI消息添加播放按钮
     */
    addPlayButtonToLastMessage: function(text) {
        const aiMessages = document.querySelectorAll('.ai-message:not(.thinking)');
        if (aiMessages.length === 0) return;

        const lastMessage = aiMessages[aiMessages.length - 1];
        const answerContent = lastMessage.querySelector('.answer-content');
        if (!answerContent) return;

        // 提取纯文本内容用于朗读
        const textToRead = text.replace(/<[^>]*>/g, '');

        // 创建并添加播放按钮
        const playButton = SpeechModule.createPlayButton();
        answerContent.appendChild(playButton);

        // 设置播放按钮事件
        SpeechModule.setupPlayButton(playButton, textToRead);
    },

    /**
     * 添加用户消息
     */
    addUserMessage: function(message) {
        const chatHistory = document.getElementById('chat-history');

        const messageElement = document.createElement('div');
        messageElement.className = 'message-container user-message';

        messageElement.innerHTML = `
            <div class="message-content">
                <div class="message-text">${message}</div>
                <div class="message-time">${ChessUtils.getCurrentTime()}</div>
            </div>
            <div class="user-avatar">
                <div class="avatar-circle">U</div>
            </div>
        `;

        chatHistory.appendChild(messageElement);
        this.scrollToBottom();
    },

    /**
     * 添加AI消息
     */
    addAIMessage: function(answermessage, thinkmessage) {
        const chatHistory = document.getElementById('chat-history');

        const messageElement = document.createElement('div');
        messageElement.className = 'message-container ai-message';

        // 替换换行符为HTML换行标签
        const answerMessage = ChessUtils.convertMarkdownToHTML(answermessage);
        const thinkMessage = thinkmessage.replace(/\n/g, '<br>');

        messageElement.innerHTML = `
            <div class="ai-avatar">
                <img src="/ai-avatar.png" alt="AI" onerror="this.src='/static/placeholder-avatar.png'">
            </div>
            <div>
                <div class="think-content">
                    <div class="think-label">思考过程：</div>
                    <div class="thinking">${thinkMessage}</div>
                </div>
                <div class="answer-content">
                    <div class="answer-label">回答：</div>
                    <div class="message-text">${answerMessage}</div>
                    <div class="message-time">${ChessUtils.getCurrentTime()}</div>
                </div>
            </div>
        `;

        chatHistory.appendChild(messageElement);
        this.scrollToBottom();
    },

    /**
     * 添加思考中的消息
     */
    addThinkingMessage: function() {
        const chatHistory = document.getElementById('chat-history');
        const messageId = 'thinking-' + Date.now();

        const messageElement = document.createElement('div');
        messageElement.id = messageId;
        messageElement.className = 'message-container ai-message thinking';

        messageElement.innerHTML = `
            <div class="ai-avatar">
                <img src="/ai-avatar.png" alt="AI" onerror="this.src='/static/placeholder-avatar.png'">
            </div>
            <div class="message-content">
                <div class="message-text">
                    <div class="thinking-indicator">
                        <span class="dot"></span>
                        <span class="dot"></span>
                        <span class="dot"></span>
                    </div>
                    <div class="thinking-text">思考中...</div>
                </div>
            </div>
        `;

        chatHistory.appendChild(messageElement);
        this.scrollToBottom();

        return messageId;
    },

    /**
     * 添加错误消息
     */
    addErrorMessage: function(message) {
        const chatHistory = document.getElementById('chat-history');

        const messageElement = document.createElement('div');
        messageElement.className = 'message-container system-message';

        messageElement.innerHTML = `
            <div class="message-content">
                <div class="message-text error-text">
                    <i class="bi bi-exclamation-triangle"></i> ${message}
                </div>
                <div class="message-time">${ChessUtils.getCurrentTime()}</div>
            </div>
        `;

        chatHistory.appendChild(messageElement);
        this.scrollToBottom();
    },

    /**
     * 滚动到底部
     */
    scrollToBottom: function() {
        const chatHistory = document.getElementById('chat-history');
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
};

// 导出聊天模块
// window.ChatModule = ChatModule;
Object.defineProperty(window, 'ChatModule', {
  value: ChatModule,
  writable: true,
  configurable: true
});