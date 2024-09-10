from ultralytics import YOLO
import torch


assert torch.cuda.is_available()


model = YOLO("yolov5l6u.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    results = model.train(
        data="./dataset/tennis_ball_detection_v6_yolov5/data.yaml", 
        epochs=100, 
        imgsz=640,
        resume=False,
        patience=20,
        batch=4,
        save=True,
        # save_period=1,
        device=0,
        pretrained=True,
        val=True,
        # workers=0,
    )