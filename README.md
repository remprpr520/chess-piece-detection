
# 国际象棋棋子检测系统

## 项目概述
---
$\qquad$基于YOLOv11的深度学习模型开发的国际象棋棋子检测系统，能够识别棋盘上的12种国际象棋棋子（黑白各6种）。系统提供Web界面，用户可上传图片并选择需要检测的棋子类别，返回带有检测框和中文标注的结果图像。

## 功能特点
---
- 🎯 高精度检测：YOLOv11模型针对国际象棋棋子优化训练
- 🖼️ 交互式界面：支持选择特定棋子类别进行检测
- 📊 可视化结果：带中文标签的检测框和分类图例
- ⚡ 高效推理：支持GPU加速（需CUDA环境）

## 支持的棋子类别
---
<table>
	<thead>
		<tr>
			<th>棋子类型</th>
			<th>中文名称</th>
			<th>类别ID</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>black_bishop</td>
			<td>黑象</td>
			<td>0</td>
		</tr>
		<tr>
			<td>black_king</td>
			<td>黑王</td>
			<td>1</td>
		</tr>
		<tr>
			<td>black_knight</td>
			<td>黑马</td>
			<td>2</td>
		</tr>
		<tr>
			<td>black_pawn</td>
			<td>黑兵</td>
			<td>3</td>
		</tr>
		<tr>
			<td>black_queen</td>
			<td>黑后</td>
			<td>4</td>
		</tr>
		<tr>
			<td>black_rook</td>
			<td>黑车</td>
			<td>5</td>
		</tr>
		<tr>
			<td>white_bishop</td>
			<td>白象</td>
			<td>6</td>
		</tr>
		<tr>
			<td>white_king</td>
			<td>白王</td>
			<td>7</td>
		</tr>
		<tr>
			<td>white_knight</td>
			<td>白马</td>
			<td>8</td>
		</tr>
		<tr>
			<td>white_pawn</td>
			<td>白兵</td>
			<td>9</td>
		</tr>
		<tr>
			<td>white_queen</td>
			<td>白后</td>
			<td>10</td>
		</tr>
		<tr>
			<td>white_rook</td>
			<td>白车</td>
			<td>11</td>
		</tr>
	</tbody>
</table>

## 技术栈
---
- **深度学习框架**: Ultralytics YOLOv11
- **后端**: FastAPI (Python)
- **前端**: HTML5 + CSS3 + JavaScript
- **图像处理**: OpenCV, Matplotlib
- **部署**: Uvicorn ASGI服务器

## 安装与运行
---
1.  **前置要求**
	- Python 3.8+
	- CUDA 11.7+ (如需GPU加速)
	- PyTorch 2.6+
	- ultralytics 8.3.99+
	- 至少8GB内存（推荐16GB）

2. **创建虚拟环境并安装依赖：**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```
3. **下载预训练模型(或使用自己训练的模型)**
4. **将模型文件(best.pt)放在`runs/detect/orthers/`目录下**
5. **运行系统**
6. **启动后端服务：**`python server.py`
7. **访问Web界面：**`http://localhost:8000`
8. **训练模型:** 如需自定义训练，使用train.py脚本

## 项目结构
---
```
chess-piece-detection/
	├── server.py            # FastAPI后端主程序
	├── train.py             # 模型训练脚本
	├── static/
	│   ├── index.html       # 前端界面
	│   └── ...              # 其他静态资源
	├── runs/
	│   └── detect/          # 训练结果和模型存储
	├── requirements.txt     # 需求文件
	├── temp/                # 临时文件目录
	└── README.md            # 项目说明文件
```

## 使用示例
---
1. 访问Web界面；
2. 在"选择要检测的棋子类别"区域勾选感兴趣的棋子；
3. 点击"选择图片"按钮上传国际象棋图片；
4. 等待处理完成后查看检测结果。

## 数据集地址：
---
**[chess-vision](https://universe.roboflow.com/multichess/chess-vision-ljby5/dataset/2)**

## 开发者
---
* 谢炜俊
* 彭思铭
* 韦祖骋
* 曾铮
