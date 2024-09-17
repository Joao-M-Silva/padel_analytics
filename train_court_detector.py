from ultralytics import YOLO
import torch


assert torch.cuda.is_available()


model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    results = model.train(
        data="./dataset/padel_court_v1_yolov8/data.yaml", 
        epochs=100, 
        imgsz=640,
        resume=False,
        patience=10,
        batch=16,
        save=True,
        save_period=1,
        device=0,
        pretrained=True,
        val=True,
    )