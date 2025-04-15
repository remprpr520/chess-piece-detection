/**
 * 检测历史相关函数
 */

/**
 * 加载检测历史数据
 */
async function loadDetectionHistory() {
    // 获取过滤条件
    const startDate = document.getElementById('detection-start-date').value;
    const endDate = document.getElementById('detection-end-date').value;
    const sessionId = document.getElementById('detection-session-id').value;

    // 获取表格主体元素
    const tableBody = document.getElementById('detection-table-body');
    // 显示加载指示器
    tableBody.innerHTML = '<tr><td colspan="4" class="text-center py-4"><div class="loader"></div></td></tr>';

    try {
        // 发送请求获取检测历史数据
        const response = await fetch('/api/stats/detection-history', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                start_date: startDate || null,
                end_date: endDate || null,
                session_id: sessionId || null,
                limit: 100
            })
        });

        const data = await response.json();

        // 如果没有数据，显示提示信息
        if (data.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="4" class="text-center py-4">没有找到匹配的记录</td></tr>';
            return;
        }

        // 清空表格
        tableBody.innerHTML = '';

        // 遍历数据并添加行
        data.forEach(record => {
            // 格式化时间戳
            const timestamp = moment(record.timestamp).format('YYYY-MM-DD HH:mm:ss');
            const sessionId = record.session_id || '未知';
            // 根据端点显示操作类型
            const endpoint = record.endpoint === '/detect' ? '棋子检测' : '棋谱生成';

            // 处理详情信息
            let details = '';
            if (record.pieces_detected && record.pieces_detected.length > 0) {
                details = `检测棋子: ${record.pieces_detected.join(', ')}`;
            }
            else {
                if (record.locations && record.locations.length > 0){
                details = `棋子位置: ${record.locations.join(', ')}`;
            }
                else{
                    details = '无详细信息';
                }
            }

            // 创建表格行
            const row = `
                <tr>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${timestamp}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${sessionId}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${endpoint}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${details}</td>
                </tr>
            `;
            tableBody.innerHTML += row;
        });
    } catch (error) {
        console.error('加载检测历史失败:', error);
        tableBody.innerHTML = '<tr><td colspan="4" class="text-center py-4 text-red-500">加载失败，请稍后重试</td></tr>';
    }
}