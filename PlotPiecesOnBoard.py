import cv2
import numpy as np
import os

# 棋盘颜色定义
LIGHT_COLOR = (235, 236, 208)  # 浅色格子
DARK_COLOR = (119, 149, 86)  # 深色格子


def create_chessboard(square_size=80):
    """创建标准8x8棋盘"""
    board_size = square_size * 8
    board = np.zeros((board_size, board_size, 3), dtype=np.uint8)

    for row in range(8):
        for col in range(8):
            color = LIGHT_COLOR if (row + col) % 2 == 0 else DARK_COLOR
            x1, y1 = col * square_size, row * square_size
            x2, y2 = x1 + square_size, y1 + square_size
            board[y1:y2, x1:x2] = color

    return board


def load_piece_images(piece_dir):
    """
    加载棋子图片

    参数:
        piece_dir: 棋子对应图片的路径
            图片格式：如黑马类型=2 => 2.png
    返回:
        piece_images: 棋子图片字典{piece_id: image}
    """
    piece_images = {}
    for filename in os.listdir(piece_dir):
        if filename.endswith(".png"):
            piece_id = filename.split(".")[0]  # 例如 "1" -> "1.png"
            img_path = os.path.join(piece_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 保留Alpha通道
            if img is not None:
                piece_images[piece_id] = img
    return piece_images


def draw_pieces(board, pieces, piece_images, square_size=80):
    """
    在棋盘上绘制棋子图片

    参数:
        board: 棋盘格，可以由create_chessboard生成
        pieces: 棋子信息(x, y, type, location, conf)
            x: x坐标
            y: y坐标
            type: 棋子类型(int)
            location: 棋子在棋盘中的位置信息，如a1、c5...
            conf: 该棋子的置信度
        piece_images: 棋子对应图片列表
        square_size: 棋盘格大小(像素)

    返回:
        board: 棋盘图片
    """
    for piece in pieces:
        if len(piece) < 5:
            continue  # 确保有足够的信息

        piece_id = str(piece[2])  # 棋子ID，对应图片文件名
        position = piece[3]  # 棋盘位置，如 "e4"

        if piece_id not in piece_images:
            continue  # 没有对应的棋子

        if position is None or position[0] not in 'abcdefgh' or position[1] not in '12345678':
            continue  # 跳过无效位置

        # 转换为棋盘坐标
        col = ord(position[0]) - ord('a')
        row = 8 - int(position[1])

        # 计算棋子放置位置 (左上角)
        x = col * square_size
        y = row * square_size

        # 获取棋子图片
        piece_img = piece_images[piece_id]

        # 调整棋子图片大小以适应格子
        piece_img = cv2.resize(piece_img, (square_size, square_size))

        # 如果图片有透明通道 (RGBA)，则进行混合
        if piece_img.shape[2] == 4:
            alpha = piece_img[:, :, 3] / 255.0
            for c in range(3):
                board[y:y + square_size, x:x + square_size, c] = (
                        alpha * piece_img[:, :, c] +
                        (1 - alpha) * board[y:y + square_size, x:x + square_size, c]
                )
        else:
            # 没有透明通道，直接覆盖
            board[y:y + square_size, x:x + square_size] = piece_img

    return board

def plot_pieces_on_board(pieces, piece_dir):
    """
    创建棋盘并绘制棋子图片

    参数:
        pieces: 棋子信息(x, y, type, location, conf)
            x: x坐标
            y: y坐标
            type: 棋子类型(int)
            location: 棋子在棋盘中的位置信息，如a1、c5...
            conf: 该棋子的置信度
        piece_dir: 棋子对应图片的路径
            图片格式：如黑马类型=2 => 2.png

    返回:
        board: 棋盘图片
    """

    # 加载棋子图片
    piece_images = load_piece_images(piece_dir)

    # 创建棋盘 (每个格子80像素)
    square_size = 80
    chessboard = create_chessboard(square_size)

    # 绘制棋子
    chessboard_with_pieces = draw_pieces(chessboard, pieces, piece_images, square_size)

    return chessboard_with_pieces

# 示例使用
if __name__ == "__main__":
    # 示例棋子列表
    # 每个元素是一个列表/元组，其中索引2是棋子ID(对应图片文件名)，索引3是棋盘位置
    pieces = [
        ['', '', '1', 'a1', ''], ['', '', '2', 'b1', ''], ['', '', '3', 'c1', ''], ['', '', '4', 'd1', ''],
    ]
    # 绘制棋子
    chessboard_with_pieces = plot_pieces_on_board(pieces=pieces, piece_dir ="static/pieces")

    # 显示结果
    cv2.imshow('Chessboard with Pieces', chessboard_with_pieces)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存图片
    cv2.imwrite('chessboard_with_pieces.png', chessboard_with_pieces)