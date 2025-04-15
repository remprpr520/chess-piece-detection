import cv2
import numpy as np
import  matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import copy
matplotlib.use('TkAgg')  # 必须在导入pyplot之前设置
# 方法1：设置matplotlib使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
class ChessboardCornerSelector:
    def __init__(self, image_path):
        # 读取图像
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # RGB转换用于matplotlib显示
        self.rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # 初始化角点和当前选择的点索引
        self.corners = np.zeros((4, 2), dtype=np.int32)
        self.current_point = 0
        self.points_selected = 0

        # 创建图形和轴
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('选择棋盘四个角点')

        # 显示图像
        self.ax.imshow(self.rgb_image)
        self.ax.set_title('点击选择棋盘四个角点 (顺序: 左上, 右上, 右下, 左下)')

        # 添加重置按钮
        self.reset_button_ax = plt.axes([0.8, 0.05, 0.1, 0.04])
        self.reset_button = Button(self.reset_button_ax, '重置')
        self.reset_button.on_clicked(self.reset)

        # 添加确认按钮
        self.confirm_button_ax = plt.axes([0.65, 0.05, 0.1, 0.04])
        self.confirm_button = Button(self.confirm_button_ax, '确认')
        self.confirm_button.on_clicked(self.confirm)

        # 添加点击事件
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        # 存储标记点
        self.markers = []


    def onclick(self, event):
        """处理鼠标点击事件"""
        if event.inaxes != self.ax:
            return

        # 记录角点位置
        if self.points_selected < 4:
            x, y = int(event.xdata), int(event.ydata)
            self.corners[self.points_selected] = [x, y]
            self.points_selected += 1

            # 绘制点
            marker = self.ax.plot(x, y, 'ro', markersize=10)[0]
            self.markers.append(marker)

            # 添加标签
            self.ax.text(x + 10, y + 10, f'{self.points_selected}', color='yellow',
                         fontsize=12, fontweight='bold')

            self.fig.canvas.draw()

            # 提示用户
            if self.points_selected == 4:
                print("已选择4个角点，点击'确认'继续")

    def reset(self, event):
        """重置所有选择的点"""
        self.points_selected = 0
        self.corners = np.zeros((4, 2), dtype=np.int32)

        # 清除所有标记
        for marker in self.markers:
            marker.remove()
        self.markers = []

        # 清除所有文本
        for text in self.ax.texts:
            text.remove()

        self.fig.canvas.draw()
        print("已重置所有点")

    def confirm(self, event):
        """确认选择的角点"""
        if self.points_selected == 4:
            plt.close(self.fig)
            print("已确认4个角点")
        else:
            print(f"请先选择4个角点 (当前已选择 {self.points_selected})")

    def get_corners(self):
        """返回用户选择的角点"""
        return self.corners


def perspective_transform(image, corners):
    """对图像进行透视变换，使棋盘变为正视图"""
    # 确保点是按照左上、右上、右下、左下的顺序
    src_points = corners.astype(np.float32)

    # 计算棋盘应该的宽度和高度
    width_top = np.sqrt(((corners[1][0] - corners[0][0]) ** 2) +
                        ((corners[1][1] - corners[0][1]) ** 2))
    width_bottom = np.sqrt(((corners[2][0] - corners[3][0]) ** 2) +
                           ((corners[2][1] - corners[3][1]) ** 2))
    width = max(int(width_top), int(width_bottom))

    height_left = np.sqrt(((corners[3][0] - corners[0][0]) ** 2) +
                          ((corners[3][1] - corners[0][1]) ** 2))
    height_right = np.sqrt(((corners[2][0] - corners[1][0]) ** 2) +
                           ((corners[2][1] - corners[1][1]) ** 2))
    height = max(int(height_left), int(height_right))

    # 确保棋盘是正方形
    size = max(width, height)

    # 定义变换后的目标点
    dst_points = np.array([
        [0, 0],  # 左上
        [size - 1, 0],  # 右上
        [size - 1, size - 1],  # 右下
        [0, size - 1]  # 左下
    ], dtype=np.float32)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # 进行透视变换
    warped = cv2.warpPerspective(image, M, (size, size))

    return warped


def transform_points_to_warped(original_points, corners):
    """
    将原图上的点转换到透视变换后的坐标系中

    参数:
    original_points - 原图上点的坐标列表，格式为 [[x1, y1], [x2, y2], ...]
    corners - 用于透视变换的四个角点

    返回:
    warped_points - 变换后坐标系中的点坐标列表
    """
    # 确保点按照左上、右上、右下、左下的顺序
    src_points = corners.astype(np.float32)

    # 计算棋盘应该的宽度和高度
    width_top = np.sqrt(((corners[1][0] - corners[0][0]) ** 2) +
                        ((corners[1][1] - corners[0][1]) ** 2))
    width_bottom = np.sqrt(((corners[2][0] - corners[3][0]) ** 2) +
                           ((corners[2][1] - corners[3][1]) ** 2))
    width = max(int(width_top), int(width_bottom))

    height_left = np.sqrt(((corners[3][0] - corners[0][0]) ** 2) +
                          ((corners[3][1] - corners[0][1]) ** 2))
    height_right = np.sqrt(((corners[2][0] - corners[1][0]) ** 2) +
                           ((corners[2][1] - corners[1][1]) ** 2))
    height = max(int(height_left), int(height_right))

    # 确保棋盘是正方形
    size = max(width, height)

    # 定义变换后的目标点
    dst_points = np.array([
        [0, 0],  # 左上
        [size - 1, 0],  # 右上
        [size - 1, size - 1],  # 右下
        [0, size - 1]  # 左下
    ], dtype=np.float32)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # 将输入点转换为适合变换的格式
    if not isinstance(original_points, np.ndarray):
        original_points = np.array(original_points)

    # 确保点是浮点数格式
    original_points = original_points.astype(np.float32)

    # 如果只有一个点，确保形状正确
    if original_points.ndim == 1:
        original_points = original_points.reshape(1, -1)

    # 对点应用透视变换
    # 需要将点转换为齐次坐标
    warped_points = []
    for point in original_points:
        # 添加齐次坐标的1
        p = np.array([point[0], point[1], 1])
        # 应用变换矩阵
        p_transformed = np.dot(M, p)
        # 归一化
        p_transformed = p_transformed / p_transformed[2]
        # 保存变换后的x和y坐标
        warped_points.append([p_transformed[0], p_transformed[1]])

    return np.array(warped_points)

def create_chess_grid(warped_image):
    """在变换后的图像上创建8x8的棋盘网格"""
    h, w = warped_image.shape[:2]
    grid = []

    # 计算每个格子的大小
    cell_size = h // 8  # 假设图像是正方形的

    # 创建8x8的网格
    for row in range(8):
        grid_row = []
        for col in range(8):
            # 计算格子的四个角点
            top_left = (col * cell_size, row * cell_size)
            top_right = ((col + 1) * cell_size, row * cell_size)
            bottom_right = ((col + 1) * cell_size, (row + 1) * cell_size)
            bottom_left = (col * cell_size, (row + 1) * cell_size)

            grid_cell = np.array([top_left, top_right, bottom_right, bottom_left])
            grid_row.append(grid_cell)
        grid.append(grid_row)

    return np.array(grid)


def visualize_grid(image, grid):
    """可视化网格检测结果"""
    vis_img = copy.deepcopy(image)

    # 绘制网格线
    for row in range(8):
        for col in range(8):
            cell = grid[row][col]
            for i in range(4):
                pt1 = tuple(cell[i].astype(int))
                pt2 = tuple(cell[(i + 1) % 4].astype(int))
                cv2.line(vis_img, pt1, pt2, (0, 0, 255), 2)

            # 在格子中心添加坐标标签
            center_x = int(np.mean([p[0] for p in cell]))
            center_y = int(np.mean([p[1] for p in cell]))
            label = f"{chr(97 + col)}{8 - row}"  # 国际象棋坐标 (a1, a2, ..., h8)
            cv2.putText(vis_img, label, (center_x - 10, center_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return vis_img


def find_chess_coordinates(warped_points, warped_image_size):
    """
    根据像素位置判断点位于哪个国际象棋格子中

    参数:
    warped_points - 变换后的点坐标列表，格式为 [[x1, y1], [x2, y2], ...]
    warped_image_size - 变换后图像的尺寸 (width, height)，假设是正方形

    返回:
    coordinates - 每个点对应的国际象棋坐标列表，如 ['a1', 'b3', ...]
    """
    coordinates = []
    size = warped_image_size  # 假设是正方形，所以width=height

    # 每个格子的大小（像素）
    cell_size = size // 8

    for point in warped_points:
        x, y = point[0], point[1]

        # 确保点在图像范围内
        if 0 <= x < size and 0 <= y < size:
            # 计算列 (a-h)
            col = int(x // cell_size)
            chess_col = chr(97 + col)  # 97是'a'的ASCII码

            # 计算行 (1-8)
            row = 7 - int(y // cell_size)  # 反转行号，因为图像坐标原点在左上
            chess_row = row + 1

            coordinates.append(f"{chess_col}{chess_row}")
        else:
            coordinates.append(None)  # 点不在图像范围内

    return coordinates

def find_board_in_window(image_path, original_points):
    """
    通过窗口自行选取角点，进行透视变换，并对棋子位置进行映射

    参数:
        image_path: 上传文件的路径
        original_points: 原始棋子点位数据(x, y, type, conf)
            x: x坐标
            y: y坐标
            type: 棋子类型(int)
            conf: 该棋子的置信度

    返回:
        chess_location: 棋子信息数组(x, y, type, location, conf)
            location: 棋子在棋盘中的位置信息，如a1、c5...
            其余同original_points
    """

    try:

        # 创建选择器并获取用户选择的角点
        selector = ChessboardCornerSelector(image_path)
        plt.show()  # 显示选择界面

        corners = selector.get_corners()
        print("选择的角点坐标：")
        print(corners)

        # 读取原始图像
        original_image = cv2.imread(image_path)

        # 对图像进行透视变换
        warped_image = perspective_transform(original_image, corners)
        warped_points = transform_points_to_warped(original_points, corners)
        print(warped_points)
        chess_coords = find_chess_coordinates(warped_points, warped_image.shape[0])
        print(chess_coords)

        # 创建棋盘网格
        grid = create_chess_grid(warped_image)

        # 可视化结果
        grid_visualization = visualize_grid(warped_image, grid)

        # 显示结果
        plt.figure(figsize=(12, 6))

        plt.subplot(121)
        plt.title("原始图像与选择的角点")
        img_with_corners = copy.deepcopy(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        for i, corner in enumerate(corners):
            cv2.circle(img_with_corners, tuple(corner), 10, (255, 0, 0), -1)
            plt.text(corner[0] + 10, corner[1] + 10, f'{i + 1}', color='yellow',
                     fontsize=12, fontweight='bold')
        plt.imshow(img_with_corners)

        plt.subplot(122)
        plt.title("透视变换后的棋盘与网格")
        plt.imshow(cv2.cvtColor(grid_visualization, cv2.COLOR_BGR2RGB))

        plt.tight_layout()
        plt.show()

        # 去除重复，选取置信度最高的
        sets = {}
        chess_location = []
        for i in range(len(original_points)):
            chess_location.append((original_points[i][0], original_points[i][1], original_points[i][2], chess_coords[i],original_points[i][3]))
            if chess_coords[i] in sets.keys():
                loc = sets[chess_coords[i]]
                if chess_location[loc][4] < original_points[i][3]:
                    chess_location[loc], chess_location[len(chess_location) - 1] = chess_location[len(chess_location) - 1], chess_location[loc]
                chess_location.pop()
            else:
                sets[chess_coords[i]] = i

        return chess_location

    except Exception as e:
        print(f"发生错误: {e}")

def find_board(image_np, corners, original_points):
    """
    没有窗口交互，直接输入角点，进行透视变换，并对棋子位置进行映射

    参数:
        image_np: numpy数组形式的图片数据
        corners: 角点信息, shape=(4, 2), 如[[100, 200], [100 ,500], [500, 200], [500, 500]]
        original_points: 原始棋子点位数据(x, y, type, conf)
            x: x坐标
            y: y坐标
            type: 棋子类型(int)
            conf: 该棋子的置信度

    返回:
        chess_location: 棋子信息数组(x, y, type, location, conf)
            location: 棋子在棋盘中的位置信息，如a1、c5...
            其余同original_points
    """
    try:
        # 验证输入图像是否为有效的numpy数组
        if not isinstance(image_np, np.ndarray) or image_np.size == 0:
            raise ValueError("输入的image_np必须是有效的非空numpy数组")

        # 使用传入的numpy图像数据（深拷贝避免污染原始数据）
        original_image = image_np.copy()

        # 对图像进行透视变换
        warped_image = perspective_transform(original_image, corners)
        warped_points = transform_points_to_warped(original_points, corners)
        chess_coords = find_chess_coordinates(warped_points, warped_image.shape[0])

    except Exception as e:
        print(f"发生错误: {e}")
        return None

    # 去除重复，选取置信度最高的
    sets = {}
    chess_location = []
    for i in range(len(original_points)):
        chess_location.append((original_points[i][0], original_points[i][1], original_points[i][2],chess_coords[i], original_points[i][3]))
        if chess_coords[i] in sets.keys():
            loc = sets[chess_coords[i]]
            if chess_location[loc][4] < original_points[i][3]:
                chess_location[loc], chess_location[len(chess_location) - 1] = chess_location[len(chess_location) - 1], chess_location[loc]
            chess_location.pop()
        else:
            sets[chess_coords[i]] = i

    return chess_location

if __name__ == "__main__":
    points = [(200, 200, 1, 0.3), (100, 200, 12, 0.5), (100, 200, 0, 0.7), (200, 200, 10, 0.8), (200, 200, 11, 0.5),  (200, 200, 8, 0.9), (200, 200, 5, 0.2)]
    ret = find_board_in_window(image_path ="test/t2.jpg", original_points = points)
    print(ret)