/**
 * 主入口文件 - 导入所有模块并初始化应用
 */

// 确保在DOM加载完成后初始化应用
document.addEventListener('DOMContentLoaded', function() {
    // 初始化UI模块
    UIModule.initializeApp();
    
    // 初始化聊天功能
    ChatModule.setupChatEvents();
    
    // 初始化语音识别
    SpeechModule.initializeSpeechRecognition();
});