/**
 * UI模块 - 处理用户界面交互和显示
 */

const UIModule = {
    /**
     * 初始化应用
     */
    initializeApp: function() {
        // 隐藏初始不需要显示的元素
        ChessUtils.hideElements(['loadingSection', 'resultSection']);
        
        // 设置标签页切换功能
        this.setupTabSwitching();
        
        // 设置棋子选择复选框监听器
        this.setupCheckboxListeners();
        
        // 设置文件上传功能
        this.setupFileUpload();
        
        // 设置棋谱生成功能
        this.setupNotationGeneration();
        
        // 更新选择计数
        this.updateSelectionCount();
    },

    /**
     * 设置页面切换功能
     */
    setupTabSwitching: function() {
        const tabs = [
            { btnId: 'detection-btn', sectionId: 'detection-section' },
            { btnId: 'qa-btn', sectionId: 'qa-section' }
        ];

        tabs.forEach(tab => {
            document.getElementById(tab.btnId).addEventListener('click', () => this.switchTab(tab.btnId, tab.sectionId));
        });
    },

    /**
     * 切换标签页
     */
    switchTab: function(activeBtnId, activeSectionId) {
        const tabs = [
            { btnId: 'detection-btn', sectionId: 'detection-section' },
            { btnId: 'qa-btn', sectionId: 'qa-section' }
        ];

        tabs.forEach(tab => {
            const btn = document.getElementById(tab.btnId);
            const section = document.getElementById(tab.sectionId);

            if (tab.btnId === activeBtnId) {
                section.style.display = 'block';
                btn.classList.remove('btn-outline-primary');
                btn.classList.add('btn-primary', 'active');
            } else {
                section.style.display = 'none';
                btn.classList.remove('btn-primary', 'active');
                btn.classList.add('btn-outline-primary');
            }
        });
    },

    /**
     * 设置复选框监听器
     */
    setupCheckboxListeners: function() {
        document.querySelectorAll('.piece-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', () => this.updateSelectionCount());
        });
    },

    /**
     * 全选/取消全选功能
     */
    toggleSelectAll: function() {
        const checkboxes = document.querySelectorAll('.piece-checkbox');
        const allChecked = Array.from(checkboxes).every(cb => cb.checked);
        checkboxes.forEach(cb => cb.checked = !allChecked);
        this.updateSelectionCount();
    },

    /**
     * 更新选择计数
     */
    updateSelectionCount: function() {
        const count = document.querySelectorAll('.piece-checkbox:checked').length;
        document.querySelector('#selectionCount').textContent = `已选择 ${count} 个类别`;
    },

    /**
     * 设置文件上传功能
     */
    setupFileUpload: function() {
        const fileInput = document.getElementById('fileInput');
        const uploadBtn = document.getElementById('uploadBtn');

        uploadBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => DetectionModule.handleFileUpload(e));
    },

    /**
     * 验证图片文件
     */
    validateImageFile: function(file) {
        if (!file.type.startsWith('image/')) {
            alert('请选择图片文件');
            return false;
        }
        return true;
    },

    /**
     * 重置文件输入框
     */
    resetFileInput: function() {
        document.getElementById('fileInput').value = '';
    },

    /**
     * 更新图片源
     */
    updateImageSource: function(imageElement, file) {
        if (imageElement.src) {
            URL.revokeObjectURL(imageElement.src);
        }
        imageElement.src = URL.createObjectURL(file);
    },

    /**
     * 设置棋谱生成功能
     */
    setupNotationGeneration: function() {
        this.setupCornerSelectionModal();

        document.getElementById('generateNotationBtn').addEventListener('click', function() {
            const sessionId = document.getElementById('resultSection').dataset.sessionId;

            if (!sessionId) {
                alert('检测会话已失效，请重新上传图片');
                return;
            }

            UIModule.openCornerSelectionModal(sessionId);
        });
    },

    /**
     * 设置角点选择模态框
     */
    setupCornerSelectionModal: function() {
        const modal = document.getElementById('cornerSelectionModal');
        const selectionImage = document.getElementById('selectionImage');

        window.modalCornerCoordinates = [];

        // 关闭按钮事件
        modal.querySelector('.close-button').addEventListener('click', () => this.closeModal(modal));

        // 点击模态框外部关闭
        window.addEventListener('click', event => {
            if (event.target === modal) {
                this.closeModal(modal);
            }
        });

        // 重置按钮事件
        document.getElementById('modalResetBtn').addEventListener('click', () => this.resetModalCornerSelection());

        // 提交按钮事件
        document.getElementById('modalSubmitBtn').addEventListener('click', () => DetectionModule.handleCornerSubmission());

        // 图片点击事件
        selectionImage.addEventListener('click', (event) => this.handleSelectionImageClick(event));
    },

    /**
     * 关闭模态框
     */
    closeModal: function(modal) {
        modal.style.display = 'none';
        modal.classList.remove('show');
    },

    /**
     * 打开角点选择模态框
     */
    openCornerSelectionModal: function(sessionId) {
        const modal = document.getElementById('cornerSelectionModal');
        modal.dataset.sessionId = sessionId;
        this.resetModalCornerSelection();
        modal.style.display = 'block';
        modal.classList.add('show');
    },

    /**
     * 处理选择图片点击事件
     */
    handleSelectionImageClick: function(event) {
        if (window.modalCornerCoordinates.length >= 4) return;

        const selectionImage = document.getElementById('selectionImage');
        const rect = selectionImage.getBoundingClientRect();

        // 计算点击位置相对于图片的坐标
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        // 计算相对于原图的比例坐标
        const naturalX = Math.round((x / rect.width) * selectionImage.naturalWidth);
        const naturalY = Math.round((y / rect.height) * selectionImage.naturalHeight);

        // 添加角点坐标
        window.modalCornerCoordinates.push({x: naturalX, y: naturalY});

        // 更新角点计数
        document.getElementById('modalCornerCount').textContent = window.modalCornerCoordinates.length;

        // 创建角点标记
        this.createModalCornerMarker(x, y, window.modalCornerCoordinates.length);

        // 如果已选择4个点，启用提交按钮
        if (window.modalCornerCoordinates.length === 4) {
            document.getElementById('modalSubmitBtn').disabled = false;
        }
    },

    /**
     * 创建模态框中的角点标记
     */
    createModalCornerMarker: function(x, y, index) {
        const container = document.querySelector('.image-selection-container');

        const marker = document.createElement('div');
        marker.className = 'corner-marker';
        marker.setAttribute('data-index', index);
        marker.style.left = `${x}px`;
        marker.style.top = `${y}px`;

        container.appendChild(marker);
    },

    /**
     * 重置模态框角点选择
     */
    resetModalCornerSelection: function() {
        window.modalCornerCoordinates = [];
        document.getElementById('modalCornerCount').textContent = '0';
        document.getElementById('modalSubmitBtn').disabled = true;
        document.querySelectorAll('.corner-marker').forEach(marker => marker.remove());
    },

    /**
     * 重置角点选择
     */
    resetCornerSelection: function() {
        document.querySelectorAll('.corner-marker').forEach(point => point.remove());
        window.cornerCoordinates = [];
        document.getElementById('modalCornerCount').textContent = '0';
        document.getElementById('modalSubmitBtn').disabled = true;
    }
};

// 导出UI模块
// window.UIModule = UIModule;
Object.defineProperty(window, 'UIModule', {
  value: UIModule,
  writable: true,
  configurable: true
});
// 全局函数，用于HTML中的onclick调用
window.toggleSelectAll = function() {
    UIModule.toggleSelectAll();
};