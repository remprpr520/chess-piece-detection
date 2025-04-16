
# 国际象棋棋子检测系统

## 项目概述
基于YOLOv11的深度学习模型开发的国际象棋棋子检测系统，能够识别棋盘上的12种国际象棋棋子（黑白各6种）。系统提供Web界面，用户可上传图片并选择需要检测的棋子类别，返回带有检测框和中文标注的结果图像。系统还支持棋谱识别和象棋知识问答功能，用户可通过点击棋盘四个角点来获取棋谱信息，并与ai进行文字和语音交流。

## 功能特点
- 🎯 高精度检测：YOLOv11模型针对国际象棋棋子优化训练
- 🖼️ 交互式界面：支持选择特定棋子类别进行检测
- 📊 可视化结果：带中文标签的检测框和分类图例
- 📜 棋谱识别：通过点击棋盘四个角点获取棋谱信息
- 🤖 知识问答：与AI进行象棋相关问题的互动
- 🗂️ 历史记录：保存用户上传的图片和检测结果
- 🔊 语音输入：支持语音与AI进行交互
- 📦 模型训练：提供自定义训练脚本，支持用户上传数据集进行模型训练
- 📈 训练可视化：提供训练过程中的损失曲线和精度曲线可视化
- 🛠️ 部署简易：使用FastAPI和Uvicorn进行后端服务部署，支持本地和云端部署

## 支持的棋子类别
<table style="text-align: center; width: 100%;">
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
- **深度学习框架**: Ultralytics YOLOv11
- **后端**: FastAPI (Python)
- **前端**: HTML5 + CSS3 + JavaScript
- **图像处理**: OpenCV, Matplotlib
- **部署**: Uvicorn ASGI服务器
- **大模型微调**: LoRA微调
- **语言模型部署**: Ollama

## 安装与运行
1. **前置要求**
    - Python 3.8+
    - CUDA 11.7+ (如需GPU加速)
    - PyTorch 2.6+
    - ultralytics 8.3.99+
    - FastAPI 0.95+
    - Uvicorn 0.22+
    - OpenCV 4.5+
    - 至少8GB内存（推荐16GB）

2. **克隆项目：**
   ```bash
   git clone https://github.com/remprpr520/chess-piece-detection.git
   ```
3. **创建虚拟环境并安装依赖：**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate  # Windows
    pip install -r requirements.txt
    ```
4. **下载预训练模型(或使用自己训练的模型)**
5. **将模型文件(best.pt)放在`runs/detect/orthers/`目录下**
6. **配置本地ollama环境**
    - 安装ollama
    - 在huggingface上下载gguf模型`ollama run hf.co/jnszstm/deepseek_r1_8b_chess_cn_gguf`
    - 修改server.py中模型名称`FIXED_MODEL_NAME  = "hf.co/jnszstm/deepseek_r1_8b_chess_cn_gguf:latest"`
7. **启动后端服务：**`python server.py`
8. **访问Web界面：**`http://localhost:8000`
9. **训练模型:** 如需自定义训练，使用train.py脚本

## 项目结构
```
chess-piece-detection/
	├── server.py                               # FastAPI后端主程序
	├── FindBoard.py                            # 棋子检测模块
	├── PlotPiecesOnBoard.py                    # 棋谱绘制模块
	├── train.py                                # 模型训练脚本
	├── static/                                 # 静态文件夹
	│   ├── dashboard/                          # dashboard页面
	│   │   ├── dashboard.html                  # 页面框架
	│   │   ├── js/                             # js文件夹
	│   │   │   ├── charts.js                   # 图表模块
	│   │   │   ├── dashboard.js                # 核心模块
	│   │   │   ├── detectionHistory.js         # 检测历史模块
	│   │   │   ├── qaHistory.js                # 历史记录模块
	│   │   │   └── utils.js                    # 工具函数模块
	│   │   └── css/                            # css文件夹
	│   │      └── styles.css                   # 样式文件
	│   ├── index/                              # 主界面
	│   │   ├── index.html                      # 主界面框架
	│   │   ├── js/                             # js文件夹
	│   │   │   ├── api.js                      # 接口脚本模块
	│   │   │   ├── chat.js                     # ai对话模块
	│   │   │   ├── detection.js                # 棋子检测模块
	│   │   │   ├── index.js                    # 核心模块
	│   │   │   ├── speech.js                   # 语音识别模块
	│   │   │   ├── ui.js                       # UI模块
	│   │   │   └── utils.js                    # 工具函数模块
	│   │   └── css/                            # css文件夹
	│   │      ├── animation.css                # 动画效果  
	│   │      ├── base.css                     # 基础样式
	│   │      ├── chat.css                     # 对话框样式
	│   │      ├── components.css               # 组件样式
	│   │      ├── modal.css                    # 模态框样式
	│   │      └── styles.css                   # 核心样式
	│   ├── pieces                              # 棋子与棋盘图片资源
	│   └── ...                                 # 其他静态资源
	├── runs/
	│   └── detect/                             # 训练结果和模型存储
	├── requirements.txt                        # 需求文件
	├── temp/                                   # 临时文件目录
	└── README.md                               # 项目说明文件
```

## 使用示例
1. 访问Web界面；
2. 在"选择要检测的棋子类别"区域勾选感兴趣的棋子；
3. 点击"选择图片"按钮上传国际象棋图片；
4. 等待处理完成后查看检测结果；
5. 点击"查看棋谱"按钮选择棋盘四个角；
6. 提交四个角点即可查看棋谱。
7. 点击"象棋知识问答"按钮，输入问题与ai互动。
8. 点击"语音输入"按钮，使用语音与ai进行交互。
9. 点击"历史记录"按钮，查看历史记录。

## 数据集地址
**[chess-vision](https://universe.roboflow.com/multichess/chess-vision-ljby5/dataset/2)**

## 代码仓库
本项目托管在 GitHub 上：  
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yourusername/your-repo)  

## 问题反馈
遇到问题？请到 [Issues](https://github.com/remprpr520/chess-piece-detection/issues) 提交，我们会尽快处理！  
⚠️ 提交前请确认是否已有类似问题。
## Star & Fork
如果觉得项目不错，请点个 ⭐ Star 支持我们！  
[![GitHub Stars](https://img.shields.io/github/stars/remprpr520/chess-piece-detection?style=social)](https://github.com/remprpr520/chess-piece-detection)

## 开发者
* **算法工程师：** 谢炜俊
  * 联系方式：QQ 2744386539
* **运维工程师：** 彭思铭
  * 联系方式：QQ 1449963079
* **前端工程师：** 韦祖骋
  * 联系方式：QQ 2249945094
* **后端工程师：** 曾铮
  * 联系方式：QQ 1846883239