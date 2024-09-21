from typing import Literal
from dataclasses import dataclass
from tqdm import tqdm
import json
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import models
from ultralytics import YOLO

from trackers.keypoints_tracker.keypoints_dataset import InferenceKeypointDataset


@dataclass
class Keypoint:
    
    id: int
    xy: tuple[float, float]

    @classmethod
    def from_dict(cls, x: dict):
        return cls(**x)
    
    def asint(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.xy)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "xy": self.xy,
        }

    def draw(self, frame: np.ndarray) -> np.ndarray:
        x, y = self.xy
        cv2.putText(
            frame, 
            str(self.id + 1),
            (int(x) + 5, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        cv2.circle(
            frame,
            (int(x), int(y)),
            radius=6,
            color=(255, 0, 0),
            thickness=-1,
        )

        return frame

class KeypointsTracker:

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUMBER_KEYPOINTS = 12
    NUMBER_EXTRA_KEYPOINTS = 6
    
    def __init__(
        self, 
        model_path: str,
        extra_model_path: str = None,
        model_type: Literal["resnet", "yolo"] = "resnet",
        fixed_keypoints_detection: list[Keypoint] = None,
    ):
        self.model_type = model_type
        if model_type == "resnet":
            self.model = models.resnet50(pretrained=True)
            self.model.fc = torch.nn.Linear(
                self.model.fc.in_features, 
                self.NUMBER_KEYPOINTS*2,
            )

            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
        elif model_type == "yolo":
            self.model = YOLO(model_path)
        else:
            raise ValueError("Unknown model type")
        
        if extra_model_path:
            self.extra_model = YOLO(extra_model_path)
        else:
            self.extra_model = None

        self.fixed_keypoints_detection = fixed_keypoints_detection

    def parse_detections(
        self, 
        detections: list[list[Keypoint]],
    ) -> list:
        parsable_detections = []
        for detection in detections:
            parsable_detections.append(
                [
                    keypoint.to_dict()
                    for keypoint in detection
                ]
            )

        return parsable_detections

    def load_detections(
        self, 
        path: str | Path,
    ) -> list[list[Keypoint]]:
        
        print("Loading Keypoint Detections ...")

        with open(path, "r") as f:
            parsable_keypoints_detections = json.load(f)

        keypoints_detections = []
        for keypoints_detection in parsable_keypoints_detections:
            keypoints_detections.append(
                [
                    Keypoint.from_dict(keypoint)
                    for keypoint in keypoints_detection
                ]
            )
        
        print("Done.")

        return keypoints_detections
    
    def detect_frame(
        self, 
        frame: np.ndarray,
        use_extra_model: bool = False,
    ) -> list[Keypoint]:

        if self.model_type != "yolo":
            raise ValueError("Only YOLO can predict frame by frame")

        if use_extra_model:
            point_mapper = {
                0: 0,
                1: 1,
                2: 2, 
                3: 3,
                4: 4, 
                5: 5,
            }
        else:
            point_mapper = {
                0: 10,
                1: 11,
                2: 1,
                3: 0,
                4: 7,
                5: 9,
                6: 8,
                7: 5,
                8: 6,
                9: 2,
                10: 4,
                11: 3,
            }

        h_frame, w_frame = frame.shape[:2] 
        train_imgsz = 640
        ratio_x = w_frame / train_imgsz
        ratio_y = h_frame / train_imgsz

        keypoints = []

        if use_extra_model:
            result = self.extra_model.predict(
                Image.fromarray(frame).resize((train_imgsz, train_imgsz)),
                conf=0.5,
                iou=0.7,
                imgsz=train_imgsz,
                device=self.DEVICE,
                max_det=self.NUMBER_EXTRA_KEYPOINTS,
            )[0]
        else:
            result = self.model.predict(
                Image.fromarray(frame).resize((train_imgsz, train_imgsz)),
                conf=0.5,
                iou=0.7,
                imgsz=train_imgsz,
                device=self.DEVICE,
                max_det=self.NUMBER_KEYPOINTS,
            )[0]

        for i, keypoint in enumerate(result.keypoints.xy.squeeze(0)):
            keypoints.append(
                Keypoint(
                    id=point_mapper[i],
                    xy=(keypoint[0].item() * ratio_x, keypoint[1].item() * ratio_y),
                )
            )

        return keypoints
            
    def detect_frames(
        self, 
        frames: list[np.ndarray],
        batch_size: int = 8,
        save_path: str | Path = None,
        load_path: str | Path = None,
        frequency: int = 1,
        use_extra_model: bool = False,
    ) -> list[list[Keypoint]]:
        
        if self.fixed_keypoints_detection is not None:
            keypoints_detections = [
                self.fixed_keypoints_detection
                for _ in range(len(frames))
            ]
            print("-"*20)
            print("USING FIXED KEYPOINTS DETECTION")
            print("-"*20)
        else:
            if load_path is not None:
                keypoints_detections = self.load_detections(load_path)
            
                return keypoints_detections
            
            self.model.to(self.DEVICE)
            if use_extra_model:
                self.extra_model.to(self.DEVICE)

            if self.model_type == "resnet":
                self.model.eval()
                dataset = InferenceKeypointDataset(frames)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                h_frame, w_frame = frames[0].shape[:2]

                keypoints_detections = []
                for batch in tqdm(loader):
                    with torch.no_grad():
                        outputs = self.model(batch["image"].to(self.DEVICE))
                        outputs = torch.nn.Sigmoid()(outputs).cpu().detach().numpy()
                    
                    for keypoints_detection in outputs:
                        keypoints_detections.append(
                            [
                                Keypoint(
                                    i,
                                    (
                                        keypoint[0] * w_frame,
                                        keypoint[1] * h_frame,
                                    )
                                )
                                for i, keypoint in enumerate(
                                    keypoints_detection.reshape(
                                        self.NUMBER_KEYPOINTS, 
                                        2,
                                    )
                                )
                            ]
                        )

            elif self.model_type == "yolo":
                keypoints_detections = []
                for i in range(0, len(frames), frequency):
                    keypoints = self.detect_frame(
                        frames[i], 
                        use_extra_model=use_extra_model,
                    )
                    keypoints_detections += [
                        keypoints 
                        for _ in range(frequency)
                    ]
                
                keypoints_detections = keypoints_detections[:len(frames)]

                #for frame in frames:
                #    keypoints = self.detect_frame(frame)
                #    keypoints_detections.append(keypoints)

        if save_path is not None and load_path is None:

            print("Saving Keypoints Detections...")

            with open(save_path, "w") as f:
                json.dump(
                    self.parse_detections(keypoints_detections),
                    f,
                )

            print("Done.")

        self.model.to("cpu")
        if use_extra_model:
            self.extra_model.to("cpu")

        return keypoints_detections

    def draw_single_frame(
        self,
        frame: np.ndarray,
        keypoints_detection: list[Keypoint],
    ) -> list[np.ndarray]:
        for keypoint in keypoints_detection:
            frame = keypoint.draw(frame)

        return frame
    
    def draw_multiple_frames(
        self, 
        frames: list[np.ndarray], 
        keypoints_detections: list[list[Keypoint]],
    ) -> list[np.ndarray]:
        output_frames = []
        for frame, keypoints_detection in zip(frames, keypoints_detections):
            frame = self.draw_single_frame(frame, keypoints_detection)
            output_frames.append(frame)

        return output_frames
    
