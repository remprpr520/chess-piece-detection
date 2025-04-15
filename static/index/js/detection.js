/**
 * 检测模块 - 处理棋子检测和棋谱生成相关功能
 */

const DetectionModule = {
    // 全局变量
    originalImageData: null,
    originalImageWidth: null,
    originalImageHeight: null,

    /**
     * 处理文件上传
     */
    handleFileUpload: async function(e) {
        const file = e.target.files[0];
        if (!file) return;

        if (!UIModule.validateImageFile(file)) {
            UIModule.resetFileInput();
            return;
        }

        const selectionImage = document.getElementById('selectionImage');
        UIModule.updateImageSource(selectionImage, file);

        const selectedPieces = this.getSelectedPieces();
        if (selectedPieces.length === 0) {
            alert('请至少选择一种棋子类型');
            return;
        }

        ChessUtils.showElement('loadingSection');
        ChessUtils.hideElement('resultSection');

        try {
            const responseData = await APIModule.detectPieces(file, selectedPieces);
            this.processDetectionResult(responseData);
        } catch (error) {
            alert(error.message);
        } finally {
            ChessUtils.hideElement('loadingSection');
            UIModule.resetFileInput();
        }
    },

    /**
     * 获取已选择的棋子类型
     */
    getSelectedPieces: function() {
        return Array.from(document.querySelectorAll('.piece-checkbox:checked'))
            .map(cb => cb.value);
    },

    /**
     * 处理检测结果
     */
    processDetectionResult: function(responseData) {
        const sessionId = responseData.session_id;
        document.getElementById('resultSection').dataset.sessionId = sessionId;
        console.log('session_id:', sessionId);

        const img = new Image();
        img.onload = function() {
            DetectionModule.setupResultCanvas(img);
            ChessUtils.showElement('resultSection');
            UIModule.resetCornerSelection();
            ChessUtils.hideElement('notationResult');
        };

        img.src = `data:image/png;base64,${responseData.image}`;
    },

    /**
     * 设置结果画布
     */
    setupResultCanvas: function(img) {
        const resultCanvas = document.getElementById('resultCanvas');
        const ctx = resultCanvas.getContext('2d');

        this.originalImageWidth = img.width;
        this.originalImageHeight = img.height;

        resultCanvas.width = img.width;
        resultCanvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        this.originalImageData = ctx.getImageData(0, 0, resultCanvas.width, resultCanvas.height);
    },

    /**
     * 处理角点提交
     */
    handleCornerSubmission: async function() {
        if (window.modalCornerCoordinates.length !== 4) {
            alert('请选择四个角点');
            return;
        }

        const modal = document.getElementById('cornerSelectionModal');
        UIModule.closeModal(modal);
        ChessUtils.showElement('loadingSection');

        try {
            const sessionId = modal.dataset.sessionId;
            const resultImage = await APIModule.generateNotation(sessionId, window.modalCornerCoordinates);
            document.getElementById('resultImage').src = URL.createObjectURL(resultImage);
            ChessUtils.showElement('notationResult');
        } catch (error) {
            alert(error.message);
        } finally {
            ChessUtils.hideElement('loadingSection');
        }
    }
};

// 导出检测模块
// window.DetectionModule = DetectionModule;
Object.defineProperty(window, 'DetectionModule', {
  value: DetectionModule,
  writable: true,
  configurable: true
});