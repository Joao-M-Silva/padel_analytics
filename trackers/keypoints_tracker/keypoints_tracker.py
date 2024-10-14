from typing import Literal, Iterable, Optional, Type
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
import supervision as sv

from trackers.keypoints_tracker.iterable import KeypointsIterable
from trackers.tracker import Object, Tracker, NoPredictFrames, NoPredictSample


class Keypoint:

    """
    Padel court keypoint detection in a given video frame

    Attributes:
        id: keypoint unique identifier
        xy: keypoint position coordinates in pixels
    """

    def __init__(self, id: int, xy: tuple[float, float]):
        self.id = id
        self.xy = xy

    @classmethod
    def from_json(cls, x: dict):
        return cls(**x)
    
    def serialize(self) -> dict:
        return {
            "id": self.id,
            "xy": self.xy,
        }
    
    def asint(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.xy)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw keypoint detection in a given frame
        """

        x, y = self.asint()

        cv2.putText(
            frame, 
            str(self.id + 1),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        cv2.circle(
            frame,
            (x, y),
            radius=6,
            color=(255, 0, 0),
            thickness=-1,
        )

        return frame
    

class Keypoints(Object):

    """
    Court multiple keypoint detection in a given video frame
    """

    def __init__(self, keypoints: list[Keypoint]):
        super().__init__()

        self.keypoints = sorted(keypoints, key=lambda x: x.id)
        self.keypoints_by_id = {
            keypoint.id: keypoint
            for keypoint in keypoints
        }

    @classmethod
    def from_json(cls, x: list[dict]) -> "Keypoints":
        return cls(
            keypoints=[
                Keypoint.from_json(keypoint_json)
                for keypoint_json in x
            ]
        )
    
    def serialize(self) -> list[dict]:
        return [
            keypoint.serialize()
            for keypoint in self.keypoints
        ]
    
    def __len__(self) -> int:
        return len(self.keypoints)
    
    def __iter__(self) -> Iterable[Keypoint]:
        return (keypoint for keypoint in self.keypoints)
    
    def __getitem__(self, id: int) -> Keypoint:
        return self.keypoints_by_id[id]

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw court keypoints detection in a given frame 
        """
        for keypoint in self.keypoints:
            frame = keypoint.draw(frame)

        return frame


class KeypointsTracker(Tracker):

    """
    Tracker of court keypoints object

    Attributes:
        model_path: model path
        model_type: the typology of model, either resnet or yolo
        fixed_keypoints_detection: court keypoints detection o a single frame.
            Useful in fixed camera scenarios.
        load_path: serializable tracker results path 
        save_path: path to save serializable tracker results
    """

    NUMBER_KEYPOINTS = 12
    TRAIN_IMAGE_SIZE = 640
    CONF=0.5
    IOU=0.7
    
    def __init__(
        self, 
        model_path: str,
        batch_size: int,
        model_type: Literal["resnet", "yolo"] = "resnet",
        fixed_keypoints_detection: Optional[Keypoints] = None,
        load_path: Optional[str | Path] = None,
        save_path: Optional[str | Path] = None,
    ):
        super().__init__(
            load_path=load_path,
            save_path=save_path,
        )

        self.batch_size = batch_size

        self.model_type = model_type
        if model_type == "resnet":
            self.model = models.resnet50(pretrained=True)
            self.model.fc = torch.nn.Linear(
                self.model.fc.in_features, 
                self.NUMBER_KEYPOINTS*2,
            )

            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        elif model_type == "yolo":
            self.model = YOLO(model_path)
        else:
            raise ValueError("Unknown model type")

        self.fixed_keypoints_detection = fixed_keypoints_detection

    def video_info_post_init(self, video_info: sv.VideoInfo) -> "KeypointsTracker":
        return self
    
    def object(self) -> Type[Object]:
        return Keypoints
    
    def draw_kwargs(self) -> dict:
        return {}
    
    def __str__(self) -> str:
        return "keypoints_tracker"
    
    def restart(self) -> None:
        self.results.restart()

    def processor(self, frame: np.ndarray) -> Image.Image:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame).resize(
            (self.TRAIN_IMAGE_SIZE, self.TRAIN_IMAGE_SIZE),
        )
    
    def to(self, device: str) -> None:
        self.model.to(device)
    
    def predict_sample(self, sample: Iterable[np.ndarray], **kwargs) -> list[Keypoints]:
        """
        Prediction over a sample of frames
        """

        if self.fixed_keypoints_detection is not None:
            print(f"{self.__str__()}: using fixed court keypoints")
            return [
                self.fixed_keypoints_detection
                for _ in range(len(sample))
            ]

        if self.model_type != "yolo":
            raise NoPredictSample()

        points_mapper = {
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

        h_frame, w_frame = sample[0].shape[:2] 
        ratio_x = w_frame / self.TRAIN_IMAGE_SIZE
        ratio_y = h_frame / self.TRAIN_IMAGE_SIZE

        sample = [
            self.processor(frame)
            for frame in sample
        ]

        results = self.model.predict(
            sample,
            conf=self.CONF,
            iou=self.IOU,
            imgsz=self.TRAIN_IMAGE_SIZE,
            device=self.DEVICE,
            max_det=self.NUMBER_KEYPOINTS,
        )

        predictions = []
        for result in results:
            keypoints = []
            for i, keypoint_detection in enumerate(result.keypoints.xy.squeeze(0)):
                keypoint = Keypoint(
                    id=points_mapper[i],
                    xy=(
                        keypoint_detection[0].item() * ratio_x,
                        keypoint_detection[1].item() * ratio_y,
                    ),
                )
                keypoints.append(keypoint)
            
            predictions.append(Keypoints(keypoints))

        return predictions
            
    def predict_frames(self, frame_generator: Iterable[np.ndarray], **kwargs) -> list[Keypoints]:
        
        if self.fixed_keypoints_detection is not None:
            print(f"{self.__str__()}: using fixed court keypoints")
            return [
                self.fixed_keypoints_detection
                for _ in frame_generator
            ]
        
        if self.model_type == "yolo":
            raise NoPredictFrames()
            
        iterable = KeypointsIterable(frame_generator)

        loader = DataLoader(
            iterable, 
            batch_size=self.batch_size, 
            shuffle=False,
            drop_last=False,
        )

        predictions = []
        for batch in tqdm(loader):
            with torch.no_grad():
                outputs = self.model(batch["image"].to(self.DEVICE))
                outputs = torch.nn.Sigmoid()(outputs).cpu().detach().numpy()
                    
                for keypoints_detection in outputs:
                    predictions.append(
                        Keypoints(
                            [
                                Keypoint(
                                    i,
                                    (
                                        keypoint[0] * iterable.w_frame,
                                        keypoint[1] * iterable.h_frame,
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
                    )

        return predictions

    
    
