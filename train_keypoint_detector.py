from ultralytics import YOLO
import torch


assert torch.cuda.is_available()


model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    results = model.train(
        data="./dataset/extra_keypoints_v1_yolov8/data.yaml", 
        epochs=100, 
        imgsz=640,
        resume=False,
        patience=20,
        batch=16,
        save=True,
        save_period=100,
        device=0,
        pretrained=True,
        val=True,
        degrees=0.0,
        translate=0.0,
        fliplr=0.0,
        mosaic=0.0,
        scale=0.0,
        erasing=0.2, # 0.0
    )