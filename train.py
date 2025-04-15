from ultralytics import YOLO
import torch

if __name__ == '__main__':

    # 加载预训练模型
    model = YOLO("yolo11n.pt")
    # 优化器参数调优
    optimizer_params = {
        "AdamW": {"betas": (0.9, 0.99), "eps": 1e-8},
        "SGD": {"momentum": 0.937, "nesterov": True}
    }
    # 训练模型
    results = model.train(
        data='data.yaml',
        epochs=20,
        imgsz=640,
        batch=16,
        workers=6,
        patience=25,
        optimizer = 'AdamW',
        lr0 = 0.001,
        lrf = 0.01,
        momentum = 0.9,
        weight_decay = 0.05,
        save_period = 10,
        hsv_h = 0.02,
        hsv_s = 0.7,
        hsv_v = 0.4,
        degrees = 45,
        flipud = 0.3,
        scale = 0.8,
        name='yolov11_chess_e20_1',
        device='cuda:0',
    )

