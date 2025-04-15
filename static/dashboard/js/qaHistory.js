/**
 * 问答历史相关函数
 */

/**
 * 加载问答历史数据
 */
async function loadQAHistory() {
    // 获取过滤条件
    const startDate = document.getElementById('qa-start-date').value;
    const endDate = document.getElementById('qa-end-date').value;

    // 获取问答列表元素
    const qaList = document.getElementById('qa-list');
    // 显示加载指示器
    qaList.innerHTML = '<div class="text-center py-4"><div class="loader"></div></div>';

    try {
        // 发送请求获取问答历史数据
        const response = await fetch('/api/stats/question-history', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                start_date: startDate || null,
                end_date: endDate || null,
                limit: 50
            })
        });

        const data = await response.json();

        // 如果没有数据，显示提示信息
        if (data.length === 0) {
            qaList.innerHTML = '<div class="text-center py-4">没有找到匹配的记录</div>';
            return;
        }

        // 清空列表
        qaList.innerHTML = '';

        // 遍历数据并添加问答项
        data.forEach(record => {
            // 格式化时间戳
            const timestamp = moment(record.timestamp).format('YYYY-MM-DD HH:mm:ss');
            const question = record.question || '未知问题';
            const answer = record.answer || '无回答';
            const answer_md = convertMarkdownToHTML(answer);

            // 创建问答项
            const qaItem = `
                <div class="border rounded-lg p-4">
                    <div class="flex justify-between items-center mb-2">
                        <span class="text-sm text-gray-500">${timestamp}</span>
                    </div>
                    <div class="mb-3">
                        <h4 class="font-medium text-gray-900">问题:</h4>
                        <p class="text-gray-700">${question}</p>
                    </div>
                    <div>
                        <h4 class="font-medium text-gray-900">回答:</h4>
                        <div class="text-gray-700">${answer_md}</div>
                    </div>
                </div>
            `;
            qaList.innerHTML += qaItem;
        });
    } catch (error) {
        console.error('加载问答历史失败:', error);
        qaList.innerHTML = '<div class="text-center py-4 text-red-500">加载失败，请稍后重试</div>';
    }
}