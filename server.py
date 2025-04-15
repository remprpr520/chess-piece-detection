import base64
import os
import io
import re
import json
import uvicorn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO
import matplotlib.font_manager as fm
import requests
from pydantic import BaseModel
import time
from threading import Timer
from uuid import uuid4
from fastapi.responses import JSONResponse
from FindBoard import find_board
from PlotPiecesOnBoard import plot_pieces_on_board
app = FastAPI(title="国际象棋棋子检测API")
os.makedirs("temp", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 缓存区域
detection_cache = {}

# 模型路径
model = YOLO('runs/detect/orthers/1/best.pt')

# Ollama API配置
OLLAMA_API_URL = "http://localhost:11434/api/generate"
FIXED_MODEL_NAME  = "hf.co/jnszstm/deepseek_r1_8b_chess_cn_gguf:latest"

# 棋子类别映射
piece_class_mapping = {
    "black_bishop": 0,
    "black_king": 1,
    "black_knight": 2,
    "black_pawn": 3,
    "black_queen": 4,
    "black_rook": 5,
    "white_bishop": 6,
    "white_king": 7,
    "white_knight": 8,
    "white_pawn": 9,
    "white_queen": 10,
    "white_rook": 11,
}

# 棋子中文名称（用于图例显示）
piece_chinese_names = {
    0: "黑象(Black Bishop)",
    1: "黑王(Black King)",
    2: "黑马(Black Knight)",
    3: "黑兵(Black Pawn)",
    4: "黑后(Black Queen)",
    5: "黑车(Black Rook)",
    6: "白象(White Bishop)",
    7: "白王(White King)",
    8: "白马(White Knight)",
    9: "白兵(White Pawn)",
    10: "白后(White Queen)",
    11: "白车(White Rook)",
}

# 颜色映射
color_map = {
    0: '#FF0000',  # black-bishop (红色)
    1: '#00FF00',  # black-king (亮绿色)
    2: '#0000FF',  # black-knight (蓝色)
    3: '#FF00FF',  # black-pawn (品红)
    4: '#FFFF00',  # black-queen (黄色)
    5: '#00FFFF',  # black-rook (青色)
    6: '#FF8000',  # white-bishop (橙色)
    7: '#8000FF',  # white-king (紫色)
    8: '#00FF80',  # white-knight (春绿色)
    9: '#FF0080',  # white-pawn (玫瑰红)
    10: '#80FF00',  # white-queen (黄绿色)
    11: '#0080FF',  # white-rook (天蓝色)
}



# 查找系统中可用的中文字体
def find_chinese_font():
    fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi', 'STXihei', 'STHeiti', 'STKaiti', 'STSong',
             'STFangsong', 'PingFang SC', 'Heiti SC', 'Songti SC', 'Arial Unicode MS', 'WenQuanYi Zen Hei',
             'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans CN', 'Hiragino Sans GB']

    for font in fonts:
        try:
            fm.findfont(font)
            return font
        except:
            continue

    # 如果找不到合适的中文字体，返回默认字体
    return 'sans-serif'

# 设置中文字体
chinese_font = find_chinese_font()


def process_image_file(file_contents: bytes) -> np.ndarray:
    """
    处理上传的图像文件，返回numpy数组格式的图像

    参数:
        file_contents: 上传文件的二进制内容

    返回:
        numpy.ndarray: 图像数组 (RGB格式)
    """
    try:
        # 使用PIL打开图像，确保处理各种格式
        image = Image.open(io.BytesIO(file_contents))

        # 转换为RGB模式（处理PNG的RGBA或单通道图像）
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 转换为numpy数组
        image_np = np.array(image)

        return image_np
    except Exception as e:
        raise ValueError(f"无法处理图像文件: {str(e)}")


def custom_plot_detection(image_np: np.ndarray, model, classes_to_show=None, conf_threshold=0.25):
    """
    存储所有检测框并根据选择绘制指定类别的检测结果，并在图片外添加图例

    参数:
        image_np: numpy数组格式的图像 (RGB)
        model: YOLO模型实例
        classes_to_show (list): 要显示的类别ID列表(None表示显示所有类)
        conf_threshold (float): 置信度阈值

    返回:
        bytes: PNG格式的图像字节流
    """
    # 创建存储不同类别框的字典
    class_boxes = defaultdict(list)
    class_names = model.names

    # 用于记录检测到的类别
    detected_classes = set()

    # 进行预测
    results = model(image_np)
    boxes = results[0].boxes

    chess_points=[]
    # 收集所有检测框并按类别存储
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x0=(x1+x2)/2
        y0=(y1+3*y2)/4
        conf = box.conf.item()
        cls_id = int(box.cls.item())

        chess_points.append((x0, y0, cls_id, conf))
        if conf < conf_threshold:
            continue

        # 如果指定了要显示的类别且当前类别不在其中，则跳过
        if classes_to_show is not None and cls_id not in classes_to_show:
            continue

        # 记录这个类别被检测到了
        detected_classes.add(cls_id)

        # 存储框信息和相关属性
        box_info = {
            'coords': (x1, y1, x2, y2),
            'conf': conf,
            'label': f"{class_names[cls_id]}: {conf:.2f}"
        }
        class_boxes[cls_id].append(box_info)

    # 设置matplotlib全局字体为支持中文的字体
    plt.rcParams['font.family'] = chinese_font

    # 创建带有两个子图的图形：左侧是检测结果，右侧是图例
    fig = plt.figure(figsize=(16, 9))

    # 设置网格，第一个子图占用左侧75%的空间用于显示检测结果
    # 第二个子图占用右侧25%的空间用于显示图例
    gs = fig.add_gridspec(1, 4)
    ax_main = fig.add_subplot(gs[0, :3])  # 图像部分占3/4
    ax_legend = fig.add_subplot(gs[0, 3])  # 图例部分占1/4

    # 显示主图像
    ax_main.imshow(image_np)
    ax_main.set_title("棋子检测结果", fontsize=14)
    ax_main.axis('off')

    # 绘制检测框
    for cls_id, boxes_list in class_boxes.items():
        # 获取当前类别的颜色
        color = color_map.get(cls_id, 'orange')

        for box in boxes_list:
            x1, y1, x2, y2 = box['coords']

            # 创建矩形框
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none')

            # 添加矩形到图像
            ax_main.add_patch(rect)


    # 设置图例区域
    ax_legend.set_title("检测棋子类别说明", fontsize=14)
    ax_legend.axis('off')

    # 只显示被检测到的类别的图例
    if detected_classes:
        # 为图例区域创建一个简单的表格效果
        y_pos = 0.9  # 起始y位置，从顶部开始
        y_step = 0.85 / max(len(detected_classes), 1)  # 每个图例项的高度
        sorted_classes = sorted(detected_classes)  # 对类别进行排序，使图例更有条理

        for i, cls_id in enumerate(sorted_classes):
            if cls_id in color_map:
                color = color_map[cls_id]
                name = piece_chinese_names.get(cls_id, f"类别 {cls_id}")
                current_y = y_pos - i * y_step

                # 绘制颜色框
                rect = patches.Rectangle(
                    (0.1, current_y - 0.03), 0.2, 0.06,
                    linewidth=1.5,
                    edgecolor='black',
                    facecolor=color)
                ax_legend.add_patch(rect)

                # 添加类别文本
                ax_legend.text(0.35, current_y, name,
                               va='center', ha='left', fontsize=12)

    else:
        # 如果没有检测到任何棋子，显示提示信息
        ax_legend.text(0.5, 0.5, "未检测到任何选中的棋子类别",
                       va='center', ha='center', fontsize=14,
                       wrap=True, color='red')

    # 在图例底部添加总结信息
    ax_legend.text(0.5, 0.05, f"总计检测到 {len(detected_classes)} 种棋子类型",
                   va='center', ha='center', fontsize=12, color='blue')

    # 添加全局标题
    fig.suptitle('国际象棋棋子检测系统', fontsize=16)

    # 调整布局
    plt.tight_layout()
    # 确保suptitle不会被裁掉
    plt.subplots_adjust(top=0.93)

    # 保存图像到内存
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close(fig)

    # 返回字节数据
    img_buf.seek(0)
    return img_buf.getvalue(), chess_points

# 定义Ollama请求和响应的模型
class OllamaRequest(BaseModel):
    prompt: str
    model: str = FIXED_MODEL_NAME
    stream: bool = False

class OllamaResponse(BaseModel):
    think_content: str
    answer_content: str



@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content


@app.post("/generate_notation")
async def generate_chess_notation(
        session_id: str = Form(...),
        corners: str = Form(...)
):

    # 从缓存获取检测结果
    cached_data = detection_cache.get(session_id)

    if not cached_data:
        raise HTTPException(status_code=404, detail="无效的会话ID或检测结果已过期")

    try:
        # 将字符串解析为Python对象
        corners_json = json.loads(corners)

        corner=[]
        for ci in corners_json:
            x = ci['x']
            y = ci['y']
            corner.append((x,y))
        corners_np=np.array(corner)

        # 验证数据格式
        if  len(corners_np) != 4:
            raise ValueError("需要4个角点坐标")

        # 获取存储的棋子坐标点和原始图像
        chess_points = cached_data["points"]
        image_np = cached_data["image"]

        # 调用find_board函数
        board_data = find_board(
            image_np=image_np,
            corners=corners_np,  # 使用转换后的NumPy数组
            original_points=chess_points
        )

        # 生成棋谱逻辑
        chessboard_with_pieces = plot_pieces_on_board(pieces=board_data,piece_dir="static/pieces")
        rgb_image = cv2.cvtColor(chessboard_with_pieces, cv2.COLOR_BGR2RGB)

        # 编码为PNG格式的字节流
        _, encoded_image = cv2.imencode(".png", rgb_image)
        image_bytes = encoded_image.tobytes()

        cv2.imwrite('chessboard_with_pieces.png', chessboard_with_pieces)

        # 返回图像响应
        return Response(content=image_bytes,media_type="image/png")

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=400, detail=f"无法处理上传的图像: {str(e)}")

@app.post("/detect")
async def detect_chess_pieces(
        file: UploadFile = File(...),
        pieces: str = Form(...)
):
    # 检查文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    # 解析选择的棋子类别
    try:
        selected_pieces = json.loads(pieces)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="棋子数据格式错误")

    # 如果没有选择任何棋子，返回错误
    if not selected_pieces:
        raise HTTPException(status_code=400, detail="请至少选择一种棋子类别")

    # 将选择的棋子名称转换为对应的类别索引
    selected_classes = []
    for piece in selected_pieces:
        if piece in piece_class_mapping:
            selected_classes.append(piece_class_mapping[piece])

    # 读取上传的图片
    try:
        contents = await file.read()
        image_np = process_image_file(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法处理上传的图像: {str(e)}")

    try:
        # 使用自定义函数进行检测
        result_image_bytes, points = custom_plot_detection(
            image_np=image_np,
            model=model,
            classes_to_show=selected_classes,
            conf_threshold=0.4
        )

        # 生成唯一会话ID
        session_id = str(uuid4())

        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # 将检测结果存入缓存（1小时有效期）
        detection_cache[session_id] = {
            "points": points,
            "image": image_bgr,
            "timestamp": time.time()
        }

        # 返回图片和会话ID

        return JSONResponse(
            content={
                "session_id": session_id,
                "image": base64.b64encode(result_image_bytes).decode('utf-8')
            },
            headers={"Content-Type": "application/json"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测过程中出错: {str(e)}")


@app.post("/chat", response_model=OllamaResponse)
async def chat_with_model(request: OllamaRequest):
    try:
        # 使用固定模型名称，不管请求中的模型是什么
        actual_model = FIXED_MODEL_NAME

        # 准备请求数据
        data = {
            "model": actual_model,
            "prompt": request.prompt,
            "stream": False
        }

        print(f"Sending request to Ollama API: {data}")

        # 发送请求到Ollama API
        response = requests.post(OLLAMA_API_URL, json=data)

        res_text=response.json().get("response")
        pattern = r'<think>(.*?)</think>(.*)'
        match = re.search(pattern, res_text, re.DOTALL)  # re.DOTALL 确保匹配换行符

        if match:
            think_content = match.group(1).strip()  # <think> 内的内容
            answer_content = match.group(2).strip()  # 剩余内容
        else:
            # 如果没有匹配到思考标签，则整个内容作为回答
            think_content = "没有提供思考过程"
            answer_content = res_text


        # 检查响应状态
        if response.status_code != 200:
            print(f"Error from Ollama API: {response.status_code}, {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Ollama API返回错误: {response.text}"
            )

        # 解析响应
        result = response.json()
        print(f"Received response from Ollama: {result}")
        return {"think_content": think_content, "answer_content": answer_content}
    except requests.RequestException as e:
        print(f"Request exception: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"无法连接到Ollama服务: {str(e)}"
        )
    except Exception as e:
        print(f"General exception: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"处理请求时出错: {str(e)}"
        )

    # 定时清理过期缓存
def clear_expired_cache():
    now = time.time()
    expired_keys = [k for k, v in detection_cache.items() if now - v["timestamp"] > 3600]
    for k in expired_keys:
        del detection_cache[k]
    # 每半时执行一次清理
    Timer(1800, clear_expired_cache).start()

if __name__ == "__main__":
    clear_expired_cache()
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)