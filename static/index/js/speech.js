/**
 * 语音模块 - 处理语音识别和合成功能
 */

const SpeechModule = {
    // 语音识别与合成相关的全局变量
    speechSynthesis: window.speechSynthesis,
    SpeechRecognition: window.SpeechRecognition || window.webkitSpeechRecognition,
    currentUtterance: null,
    recognition: null,

    /**
     * 初始化语音识别
     */
    initializeSpeechRecognition: function() {
        const micButton = document.getElementById('mic-button');
        const recognitionStatus = document.getElementById('recognition-status');
        const questionInput = document.getElementById('question-input');

        if (!this.SpeechRecognition) {
            console.warn('您的浏览器不支持语音识别功能');
            if (micButton) {
                micButton.disabled = true;
                micButton.title = '您的浏览器不支持语音识别';
            }
            return;
        }

        this.recognition = new this.SpeechRecognition();
        this.recognition.continuous = false;
        this.recognition.interimResults = false;
        this.recognition.lang = 'zh-CN';

        this.recognition.onstart = function() {
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

        this.recognition.onend = function() {
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

        this.recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            if (questionInput) {
                questionInput.value = transcript;
                questionInput.focus();
            }
        };

        this.recognition.onerror = function(event) {
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

            ChatModule.addErrorMessage(errorMessage);
        };

        if (micButton) {
            micButton.addEventListener('click', () => this.toggleSpeechRecognition());
        }
    },

    /**
     * 切换语音识别
     */
    toggleSpeechRecognition: function() {
        if (this.recognition) {
            try {
                const micButton = document.getElementById('mic-button');
                if (micButton && micButton.classList.contains('active')) {
                    this.recognition.stop();
                } else {
                    this.recognition.start();
                }
            } catch (error) {
                console.error('语音识别错误:', error);
                ChatModule.addErrorMessage('语音识别启动失败');
            }
        }
    },

    /**
     * 语音合成功能
     */
    speakText: function(text, button) {
        if (!this.speechSynthesis) {
            console.warn('您的浏览器不支持语音合成功能');
            return;
        }

        this.stopSpeaking();

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
            SpeechModule.currentUtterance = null;
        };

        utterance.onerror = function(event) {
            console.error('语音合成错误:', event);
            if (button) {
                button.classList.remove('playing');
                button.innerHTML = '<i class="fas fa-volume-up"></i>';
            }
            SpeechModule.currentUtterance = null;
        };

        this.currentUtterance = utterance;
        this.speechSynthesis.speak(utterance);
    },

    /**
     * 停止语音播放
     */
    stopSpeaking: function() {
        if (this.speechSynthesis && this.speechSynthesis.speaking) {
            this.speechSynthesis.cancel();
            this.currentUtterance = null;
        }
    },

    /**
     * 设置播放按钮事件
     */
    setupPlayButton: function(button, text) {
        button.addEventListener('click', function() {
            if (SpeechModule.currentUtterance && SpeechModule.speechSynthesis.speaking) {
                SpeechModule.stopSpeaking();
                button.classList.remove('playing');
                button.innerHTML = '<i class="fas fa-volume-up"></i>';
            } else {
                SpeechModule.speakText(text, button);
            }
        });
    },

    /**
     * 创建播放按钮
     */
    createPlayButton: function() {
        const playButton = document.createElement('button');
        playButton.className = 'btn-play';
        playButton.innerHTML = '<i class="fas fa-volume-up"></i>';
        playButton.title = '播放语音';
        return playButton;
    },

    /**
     * 创建语音识别状态指示器
     */
    createRecognitionStatus: function() {
        const statusElement = document.createElement('div');
        statusElement.id = 'recognition-status';
        statusElement.className = 'recognition-status hidden';
        statusElement.innerHTML = `
            <div class="pulse"></div>
            <span>正在聆听...</span>
        `;
        return statusElement;
    },

    /**
     * 创建麦克风按钮
     */
    createMicButton: function() {
        const button = document.createElement('button');
        button.className = 'btn btn-mic';
        button.id = 'mic-button';
        button.type = 'button';
        button.innerHTML = '<i class="fas fa-microphone mic-icon"></i>';
        button.title = '语音输入';
        return button;
    }
};

// 在页面关闭时清理语音相关资源
window.addEventListener('beforeunload', () => {
    SpeechModule.stopSpeaking();
    if (SpeechModule.recognition) {
        SpeechModule.recognition.stop();
    }
});

// 导出语音模块
// window.SpeechModule = SpeechModule;
Object.defineProperty(window, 'SpeechModule', {
  value: SpeechModule,
  writable: true,
  configurable: true
});