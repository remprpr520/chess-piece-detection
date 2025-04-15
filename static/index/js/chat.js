// 发送问题到后端
let isProcessingQuestion = false; // 防止重复提交
// 语音识别与合成相关的全局变量
const speechSynthesis = window.speechSynthesis;
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let currentUtterance = null;
let recognition = null;

// 页面加载时初始化聊天功能
document.addEventListener('DOMContentLoaded', function() {
    // 设置聊天相关的事件
    setupChatEvents();

    // 初始化语音识别
    initializeSpeechRecognition();

    // 添加麦克风按钮到UI
    addSpeechUIElements();
});

// 添加语音相关的UI元素
function addSpeechUIElements() {
    const questionContainer = document.querySelector('.question-container');
    if (!questionContainer) return;

    // 添加麦克风按钮
    const micButton = createMicButton();
    micButton.id = 'mic-button';
    questionContainer.appendChild(micButton);

    // 添加语音识别状态指示器
    const recognitionStatus = createRecognitionStatus();
    questionContainer.appendChild(recognitionStatus);
}

// 设置聊天相关的事件
function setupChatEvents() {
    // 防止重复绑定事件
    const sendButton = document.getElementById('send-question');
    if (sendButton && !sendButton.hasEventListener) {
        sendButton.addEventListener('click', sendQuestion);
        sendButton.hasEventListener = true;
    }

    // 问题输入框回车事件
    const questionInput = document.getElementById('question-input');
    if (questionInput && !questionInput.hasEventListener) {
        questionInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendQuestion();
            }
        });
        questionInput.hasEventListener = true;
    }

    // 快速问题按钮点击事件
    document.querySelectorAll('.quick-question').forEach(button => {
        if (!button.hasEventListener) {
            button.addEventListener('click', function() {
                document.getElementById('question-input').value = this.textContent;
                sendQuestion();
            });
            button.hasEventListener = true;
        }
    });
}

function sendQuestion() {
  // 防止重复提交
  if (isProcessingQuestion) return;

  const questionInput = document.getElementById('question-input');
  const question = questionInput.value.trim();

  if (question === '') return;

  // 设置正在处理标志
  isProcessingQuestion = true;

  // 显示用户问题
  addUserMessage(question);

  // 清空输入框
  questionInput.value = '';

  // 显示AI"思考中"的消息
  const thinkingId = addThinkingMessage();

  // 添加关于国际象棋的上下文增强问题
  const enhancedPrompt = `你是一个国际象棋专家助手。请回答以下关于国际象棋的问题：${question}`;

  // 发送请求到后端 - 使用Ollama模型
  fetch('/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prompt: enhancedPrompt,
      model: 'ollama',
      stream: false
    })
  })
  .then(response => {
    if (!response.ok) {
      throw new Error('请求失败: ' + response.statusText);
    }
    return response.json();
  })
  .then(data => {

    // 移除思考中的消息
    document.getElementById(thinkingId).remove();

    // 显示AI回答
    addAIMessage(data.answer_content, data.think_content);

    // 为AI回答添加播放按钮
    addPlayButtonToLastMessage(data.answer_content);

    // 重置处理标志
    isProcessingQuestion = false;
  })
  .catch(error => {
    console.error('发送问题失败:', error);

    // 移除思考中的消息
    document.getElementById(thinkingId).remove();

    // 显示错误消息
    addErrorMessage('发送问题失败: ' + error.message);

    // 重置处理标志
    isProcessingQuestion = false;
  });
}

// 为最后一条AI消息添加播放按钮
function addPlayButtonToLastMessage(text) {
    const aiMessages = document.querySelectorAll('.ai-message:not(.thinking)');
    if (aiMessages.length === 0) return;

    const lastMessage = aiMessages[aiMessages.length - 1];
    const answerContent = lastMessage.querySelector('.answer-content');
    if (!answerContent) return;

    // 提取纯文本内容用于朗读
    const textToRead = text.replace(/<[^>]*>/g, '');

    // 创建并添加播放按钮
    const playButton = createPlayButton();
    answerContent.appendChild(playButton);

    // 设置播放按钮事件
    setupPlayButton(playButton, textToRead);
}

// 添加用户消息
function addUserMessage(message) {
  const chatHistory = document.getElementById('chat-history');

  const messageElement = document.createElement('div');
  messageElement.className = 'message-container user-message';

  messageElement.innerHTML = `
    <div class="message-content">
      <div class="message-text">${message}</div>
      <div class="message-time">${getCurrentTime()}</div>
    </div>
    <div class="user-avatar">
      <div class="avatar-circle">U</div>
    </div>
  `;

  chatHistory.appendChild(messageElement);
  scrollToBottom();
}

// 添加AI消息
function addAIMessage(answermessage, thinkmessage) {
  const chatHistory = document.getElementById('chat-history');

  const messageElement = document.createElement('div');
  messageElement.className = 'message-container ai-message';

  // 替换换行符为HTML换行标签
  const answerMessage = convertMarkdownToHTML(answermessage);
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
            <div class="message-time">${getCurrentTime()}</div>
        </div>
    </div>
  `;

  chatHistory.appendChild(messageElement);
  scrollToBottom();
}

// 添加思考中的消息
function addThinkingMessage() {
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
  scrollToBottom();

  return messageId;
}

// 添加错误消息
function addErrorMessage(message) {
  const chatHistory = document.getElementById('chat-history');

  const messageElement = document.createElement('div');
  messageElement.className = 'message-container system-message';

  messageElement.innerHTML = `
    <div class="message-content">
      <div class="message-text error-text">
        <i class="bi bi-exclamation-triangle"></i> ${message}
      </div>
      <div class="message-time">${getCurrentTime()}</div>
    </div>
  `;

  chatHistory.appendChild(messageElement);
  scrollToBottom();
}

// 获取当前时间
function getCurrentTime() {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// 滚动到底部
function scrollToBottom() {
  const chatHistory = document.getElementById('chat-history');
  chatHistory.scrollTop = chatHistory.scrollHeight;
}



// 初始化语音识别
function initializeSpeechRecognition() {
    const micButton = document.getElementById('mic-button');
    const recognitionStatus = document.getElementById('recognition-status');
    const questionInput = document.getElementById('question-input');

    if (!SpeechRecognition) {
        console.warn('您的浏览器不支持语音识别功能');
        if (micButton) {
            micButton.disabled = true;
            micButton.title = '您的浏览器不支持语音识别';
        }
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'zh-CN';

    recognition.onstart = function() {
        if (recognitionStatus) {
            recognitionStatus.classList.remove('hidden');
        }
        if (micButton) {
            micButton.classList.add('active');
            const micIcon = micButton.querySelector('.mic-icon');
            if (micIcon) {
                micIcon.className = 'fas fa-microphone-slash mic-icon';
            }
        }
    };

    recognition.onend = function() {
        if (recognitionStatus) {
            recognitionStatus.classList.add('hidden');
        }
        if (micButton) {
            micButton.classList.remove('active');
            const micIcon = micButton.querySelector('.mic-icon');
            if (micIcon) {
                micIcon.className = 'fas fa-microphone mic-icon';
            }
        }
    };

    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        if (questionInput) {
            questionInput.value = transcript;
            questionInput.focus();
        }
    };

    recognition.onerror = function(event) {
        console.error('语音识别错误:', event.error);
        if (recognitionStatus) {
            recognitionStatus.classList.add('hidden');
        }
        if (micButton) {
            micButton.classList.remove('active');
            const micIcon = micButton.querySelector('.mic-icon');
            if (micIcon) {
                micIcon.className = 'fas fa-microphone mic-icon';
            }
        }

        let errorMessage = '语音识别错误';
        if (event.error === 'not-allowed') {
            errorMessage = '请允许访问麦克风';
        } else if (event.error === 'no-speech') {
            errorMessage = '没有检测到语音';
        }

        addErrorMessage(errorMessage);
    };

    if (micButton) {
        micButton.addEventListener('click', toggleSpeechRecognition);
    }
}

// 切换语音识别
function toggleSpeechRecognition() {
    if (recognition) {
        try {
            const micButton = document.getElementById('mic-button');
            if (micButton && micButton.classList.contains('active')) {
                recognition.stop();
            } else {
                recognition.start();
            }
        } catch (error) {
            console.error('语音识别错误:', error);
            addErrorMessage('语音识别启动失败');
        }
    }
}

// 语音合成功能
function speakText(text, button) {
    if (!speechSynthesis) {
        console.warn('您的浏览器不支持语音合成功能');
        return;
    }

    stopSpeaking();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'zh-CN';
    utterance.rate = 0.9;
    utterance.pitch = 1.0;

    utterance.onstart = function() {
        if (button) {
            button.classList.add('playing');
            button.innerHTML = '<i class="fas fa-stop"></i>';
        }
    };

    utterance.onend = function() {
        if (button) {
            button.classList.remove('playing');
            button.innerHTML = '<i class="fas fa-volume-up"></i>';
        }
        currentUtterance = null;
    };

    utterance.onerror = function(event) {
        console.error('语音合成错误:', event);
        if (button) {
            button.classList.remove('playing');
            button.innerHTML = '<i class="fas fa-volume-up"></i>';
        }
        currentUtterance = null;
    };

    currentUtterance = utterance;
    speechSynthesis.speak(utterance);
}

function stopSpeaking() {
    if (speechSynthesis && speechSynthesis.speaking) {
        speechSynthesis.cancel();
        currentUtterance = null;
    }
}

function setupPlayButton(button, text) {
    button.addEventListener('click', function() {
        if (currentUtterance && speechSynthesis.speaking) {
            stopSpeaking();
            button.classList.remove('playing');
            button.innerHTML = '<i class="fas fa-volume-up"></i>';
        } else {
            speakText(text, button);
        }
    });
}

// 播放按钮相关的HTML结构
function createPlayButton() {
    const playButton = document.createElement('button');
    playButton.className = 'btn-play';
    playButton.innerHTML = '<i class="fas fa-volume-up"></i>';
    playButton.title = '播放语音';
    return playButton;
}

// 语音识别状态指示器相关HTML
function createRecognitionStatus() {
    const statusElement = document.createElement('div');
    statusElement.id = 'recognition-status';
    statusElement.className = 'recognition-status hidden';
    statusElement.innerHTML = `
        <div class="pulse"></div>
        <span>正在聆听...</span>
    `;
    return statusElement;
}

// 麦克风按钮HTML
function createMicButton() {
    const button = document.createElement('button');
    button.className = 'btn btn-mic';
    button.id = 'mic-button';
    button.type = 'button';
    button.innerHTML = '<i class="fas fa-microphone mic-icon"></i>';
    button.title = '语音输入';
    return button;
}

// 在页面关闭时清理语音相关资源
window.addEventListener('beforeunload', () => {
    stopSpeaking();
    if (recognition) {
        recognition.stop();
    }
});