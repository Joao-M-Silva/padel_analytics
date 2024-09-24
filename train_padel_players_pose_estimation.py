from ultralytics import YOLO
import torch


assert torch.cuda.is_available()


model = YOLO("yolov8m-pose.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    results = model.train(
        data="./dataset/padel_players_pose_estimation_v2_yolov8/data.yaml", 
        epochs=100, 
        imgsz=1280,
        resume=False,
        patience=20,
        batch=4,
        save=True,
        save_period=-1,  # specified in epochs
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