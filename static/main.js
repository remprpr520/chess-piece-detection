// 全局变量
let originalImageData, originalImageWidth, originalImageHeight;

// 页面加载初始化
document.addEventListener('DOMContentLoaded', initializeApp);

function initializeApp() {
    hideElements(['loadingSection', 'resultSection']);
    setupTabSwitching();
    setupCheckboxListeners();
    setupFileUpload();
    setupNotationGeneration();
    updateSelectionCount();
}

// 隐藏指定元素
function hideElements(elementIds) {
    elementIds.forEach(id => document.getElementById(id).classList.add('hidden'));
}

// 设置页面切换功能
function setupTabSwitching() {
    const tabs = [
        { btnId: 'detection-btn', sectionId: 'detection-section' },
        { btnId: 'qa-btn', sectionId: 'qa-section' }
    ];

    tabs.forEach(tab => {
        document.getElementById(tab.btnId).addEventListener('click', () => switchTab(tab.btnId, tab.sectionId));
    });
}

// 切换标签页
function switchTab(activeBtnId, activeSectionId) {
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
}

// 设置复选框监听器
function setupCheckboxListeners() {
    document.querySelectorAll('.piece-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', updateSelectionCount);
    });
}

// 全选/取消全选功能
function toggleSelectAll() {
    const checkboxes = document.querySelectorAll('.piece-checkbox');
    const allChecked = Array.from(checkboxes).every(cb => cb.checked);
    checkboxes.forEach(cb => cb.checked = !allChecked);
    updateSelectionCount();
}

// 更新选择计数
function updateSelectionCount() {
    const count = document.querySelectorAll('.piece-checkbox:checked').length;
    document.querySelector('#selectionCount').textContent = `已选择 ${count} 个类别`;
}

// 设置文件上传功能
function setupFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');

    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileUpload);
}

// 处理文件上传
async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!validateImageFile(file)) {
        resetFileInput();
        return;
    }

    const selectionImage = document.getElementById('selectionImage');
    updateImageSource(selectionImage, file);

    const selectedPieces = getSelectedPieces();
    if (selectedPieces.length === 0) {
        alert('请至少选择一种棋子类型');
        return;
    }

    showElement('loadingSection');
    hideElement('resultSection');

    try {
        const responseData = await detectPieces(file, selectedPieces);
        processDetectionResult(responseData);
    } catch (error) {
        alert(error.message);
    } finally {
        hideElement('loadingSection');
        resetFileInput();
    }
}

// 验证图片文件
function validateImageFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('请选择图片文件');
        return false;
    }
    return true;
}

// 重置文件输入框
function resetFileInput() {
    document.getElementById('fileInput').value = '';
}

// 更新图片源
function updateImageSource(imageElement, file) {
    if (imageElement.src) {
        URL.revokeObjectURL(imageElement.src);
    }
    imageElement.src = URL.createObjectURL(file);
}

// 获取已选择的棋子类型
function getSelectedPieces() {
    return Array.from(document.querySelectorAll('.piece-checkbox:checked'))
        .map(cb => cb.value);
}

// 显示元素
function showElement(elementId) {
    document.getElementById(elementId).classList.remove('hidden');
}

// 隐藏元素
function hideElement(elementId) {
    document.getElementById(elementId).classList.add('hidden');
}

// 发送检测请求
async function detectPieces(file, selectedPieces) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('pieces', JSON.stringify(selectedPieces));

    const response = await fetch('/detect', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) throw new Error('检测失败');
    return await response.json();
}

// 处理检测结果
function processDetectionResult(responseData) {
    const sessionId = responseData.session_id;
    document.getElementById('resultSection').dataset.sessionId = sessionId;
    console.log('session_id:', sessionId);

    const img = new Image();
    img.onload = function() {
        setupResultCanvas(img);
        showElement('resultSection');
        resetCornerSelection();
        hideElement('notationResult');
    };

    img.src = `data:image/png;base64,${responseData.image}`;
}

// 设置结果画布
function setupResultCanvas(img) {
    const resultCanvas = document.getElementById('resultCanvas');
    const ctx = resultCanvas.getContext('2d');

    originalImageWidth = img.width;
    originalImageHeight = img.height;

    resultCanvas.width = img.width;
    resultCanvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    originalImageData = ctx.getImageData(0, 0, resultCanvas.width, resultCanvas.height);
}

// 重置角点选择
function resetCornerSelection() {
    document.querySelectorAll('.corner-marker').forEach(point => point.remove());
    window.cornerCoordinates = [];
    document.getElementById('modalCornerCount').textContent = '0';
    document.getElementById('modalSubmitBtn').disabled = true;
}

// 设置棋谱生成功能
function setupNotationGeneration() {
    setupCornerSelectionModal();

    document.getElementById('generateNotationBtn').addEventListener('click', function() {
        const sessionId = document.getElementById('resultSection').dataset.sessionId;

        if (!sessionId) {
            alert('检测会话已失效，请重新上传图片');
            return;
        }

        openCornerSelectionModal(sessionId);
    });
}

// 设置角点选择模态框
function setupCornerSelectionModal() {
    const modal = document.getElementById('cornerSelectionModal');
    const selectionImage = document.getElementById('selectionImage');

    window.modalCornerCoordinates = [];

    // 关闭按钮事件
    modal.querySelector('.close-button').addEventListener('click', () => closeModal(modal));

    // 点击模态框外部关闭
    window.addEventListener('click', event => {
        if (event.target === modal) {
            closeModal(modal);
        }
    });

    // 重置按钮事件
    document.getElementById('modalResetBtn').addEventListener('click', resetModalCornerSelection);

    // 提交按钮事件
    document.getElementById('modalSubmitBtn').addEventListener('click', handleCornerSubmission);

    // 图片点击事件
    selectionImage.addEventListener('click', handleSelectionImageClick);
}

// 关闭模态框
function closeModal(modal) {
    modal.style.display = 'none';
}

// 打开角点选择模态框
function openCornerSelectionModal(sessionId) {
    const modal = document.getElementById('cornerSelectionModal');
    modal.dataset.sessionId = sessionId;
    resetModalCornerSelection();
    modal.style.display = 'block';
}

// 处理角点提交
async function handleCornerSubmission() {
    if (window.modalCornerCoordinates.length !== 4) {
        alert('请选择四个角点');
        return;
    }

    const modal = document.getElementById('cornerSelectionModal');
    closeModal(modal);
    showElement('loadingSection');

    try {
        const sessionId = modal.dataset.sessionId;
        const resultImage = await generateNotation(sessionId, window.modalCornerCoordinates);
        document.getElementById('resultImage').src = URL.createObjectURL(resultImage);
        showElement('notationResult');
    } catch (error) {
        alert(error.message);
    } finally {
        hideElement('loadingSection');
    }
}

// 生成棋谱
async function generateNotation(sessionId, corners) {
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
}

// 处理选择图片点击事件
function handleSelectionImageClick(event) {
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
    createModalCornerMarker(x, y, window.modalCornerCoordinates.length);

    // 如果已选择4个点，启用提交按钮
    if (window.modalCornerCoordinates.length === 4) {
        document.getElementById('modalSubmitBtn').disabled = false;
    }
}

// 创建模态框中的角点标记
function createModalCornerMarker(x, y, index) {
    const container = document.querySelector('.image-selection-container');

    const marker = document.createElement('div');
    marker.className = 'corner-marker';
    marker.setAttribute('data-index', index);
    marker.style.left = `${x}px`;
    marker.style.top = `${y}px`;

    container.appendChild(marker);
}

// 重置模态框角点选择
function resetModalCornerSelection() {
    window.modalCornerCoordinates = [];
    document.getElementById('modalCornerCount').textContent = '0';
    document.getElementById('modalSubmitBtn').disabled = true;
    document.querySelectorAll('.corner-marker').forEach(marker => marker.remove());
}