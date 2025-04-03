from ultralytics import YOLO
import cv2
import matplotlib

def yolo_image_detection(image_path):

    # 加载YOLO模型
    model = YOLO("runs/detect/yolov11_chess_e20_1/weights/best.pt")

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        return

    # 进行目标检测
    results = model(
        image,
        augment=True,
        show=False,
        verbose=False,
    )

    # 在图像上绘制检测结果
    annotated_image = results[0].plot()

    # 显示结果
    cv2.imshow("YOLO Detection", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 使用示例
if __name__ == "__main__":
    image_path = r"test/t3.jpg"  # 图像路径
    yolo_image_detection(image_path)