from typing import Literal, Iterable, Optional, Type
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import supervision as sv

from trackers.tracker import Object, Tracker, NoPredictFrames


@dataclass
class PlayerKeypoint:

    """
    Player pose keypoint detection in a given video frame

    Attributes:
        id: keypoint id
        name: keypoint unique identifier
        xy: keypoint location in the video frame
    """

    id: int
    name: str
    xy: tuple[float, float]

    def asint(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.xy)
    
    @classmethod
    def from_json(cls, x: dict):
        return cls(**x)
    
    def serialize(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "xy": self.xy,
        }
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw player pose keypoint detection in a given frame
        """
        cv2.circle(
            frame,
            self.asint(),
            radius=2,
            color=(255, 0, 0),
            thickness=-1,
        )

        return frame
    
    
class PlayerKeypoints:

    """
    Player collection of pose keypoints
    """

    KEYPOINTS_NAMES = [
        "left_foot",
        "right_foot",
        "torso",
        "right_shoulder",
        "left_shoulder",
        "head",
        "neck",
        "left_hand",
        "right_hand",
        "right_knee",
        "left_knee",
        "right_elbow",
        "left_elbow",
    ]

    CONNECTIONS = [
        ("left_foot", "left_knee"),
        ("left_knee", "torso"),
        ("right_foot", "right_knee"),
        ("right_knee", "torso"),
        ("torso", "left_shoulder"),
        ("torso", "right_shoulder"),
        ("left_hand", "left_elbow"),
        ("left_elbow", "left_shoulder"),
        ("left_shoulder", "neck"),
        ("neck", "head"),
        ("right_hand", "right_elbow"),
        ("right_elbow", "right_shoulder"),
        ("right_shoulder", "neck"),
    ]

    def __init__(self, player_keypoints: list[PlayerKeypoint]):

        self.player_keypoints = player_keypoints
        
        if player_keypoints == []:
            self.keypoints_by_name = {}
        else:
            self.keypoints_by_name = {
                keypoint.name: keypoint
                for keypoint in player_keypoints
            }

    @classmethod
    def from_json(cls, x: dict):
        player_keypoints = [
            PlayerKeypoint.from_json(keypoint)
            for keypoint in x["player_keypoints"]
        ]
        return cls(player_keypoints)
        
    def serialize(self) -> dict:
        return {
            "player_keypoints": [ 
                keypoint.serialize()
                for keypoint in self.player_keypoints
            ]
        }
    
    def __len__(self) -> int:
        return len(self.player_keypoints)
    
    def __iter__(self) -> Iterable[PlayerKeypoint]:
        return (keypoint for keypoint in self.player_keypoints)
    
    def __getitem__(self, name: str) -> PlayerKeypoint:
        
        assert name in self.KEYPOINTS_NAMES

        return self.keypoints_by_name[name]
    
    def draw(self, frame: np.ndarray) -> np.ndarray:

        """
        Draw a straight line in-between unique player keypoint connections
        """

        keypoints = {
            keypoint.name: keypoint.asint()
            for keypoint in self.player_keypoints
        }

        if keypoints == {}:
            return frame

        frame = frame.copy()

        for connection in self.CONNECTIONS:
            cv2.line(
                frame, 
                keypoints[connection[0]],
                keypoints[connection[1]],
                color=(255, 0, 0),
                thickness=2,
            )

        return frame
    
    
class PlayersKeypoints(Object):

    """
    Players pose keypoints detections in a given video frame
    """

    def __init__(self, players_keypoints: list[PlayerKeypoints]) -> None:
        super().__init__()
        self.players_keypoints = players_keypoints

    @classmethod
    def from_json(cls, x: dict | list[dict]) -> "PlayersKeypoints":
        return cls(
            players_keypoints=[
                PlayerKeypoints.from_json(player_keypoints_json)
                for player_keypoints_json in x
            ]
        )

    def serialize(self) -> list[dict]:
        return [
            player_keypoints.serialize()
            for player_keypoints in self.players_keypoints
        ]
    
    def __len__(self) -> int:
        return len(self.players_keypoints)
    
    def __iter__(self) -> Iterable[PlayerKeypoints]:
        return (player_keypoints for player_keypoints in self.players_keypoints)

    def __getitem__(self, i: int) -> PlayerKeypoints:
        return self.players_keypoints[i]
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        
        for player_keypoints in self.players_keypoints:
            frame = player_keypoints.draw(frame)

        return frame
    
    
class PlayerKeypointsTracker(Tracker):

    """
    Tracker of players keypoints object

    Attributes:
        model_path: players keypoints tracker model path
        train_image_size: image size used in training.
                          Options:
                            1. 640 for 640x640 images
                            2. 1280 for 1280x1280 images
        load_path: serializable tracker results path 
        save_path: path to save serializable tracker results
    """

    CONF = 0.25
    IOU = 0.7

    def __init__(
        self, 
        model_path: str, 
        train_image_size: Literal[640, 1280],
        batch_size: int,
        load_path: Optional[str | Path],
        save_path: Optional[str | Path],
    ):
        super().__init__(
            load_path=load_path,
            save_path=save_path,
        )

        self.model = YOLO(model_path)

        assert train_image_size in (640, 1280)

        self.train_image_size = train_image_size
        self.batch_size = batch_size

    def video_info_post_init(self, video_info: sv.VideoInfo) -> "PlayerKeypointsTracker":
        return self
    
    def object(self) -> Type[Object]:
        return PlayersKeypoints
    
    def draw_kwargs(self) -> dict:
        return {}
    
    def __str__(self) -> str:
        return "players_keypoints_tracker"
    
    def restart(self) -> None:
        self.results.restart()

    def processor(self, frame: np.ndarray) -> Image:
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return Image.fromarray(frame_rgb).resize(
            (self.train_image_size, self.train_image_size),
        )
    
    def to(self, device: str) -> None:
        self.model.to(device)

    def predict_sample(self, sample: Iterable[np.ndarray], **kwargs) -> list[PlayersKeypoints]:
        """
        Prediction over a sample of frames
        """

        h_frame, w_frame = sample[0].shape[:2]
        ratio_x = w_frame / self.train_image_size
        ratio_y = h_frame / self.train_image_size

        sample = [
            self.processor(frame)
            for frame in sample
        ]
        
        results = self.model.predict(
            sample,
            conf=self.CONF,
            iou=self.IOU,
            imgsz=self.train_image_size,
            device=self.DEVICE,
            classes=[0],
        )

        predictions = []
        for result in results:

            players_keypoints = [] 

            players_keypoints_detection = result.keypoints.xy.squeeze(0)
            if len(players_keypoints_detection) == 2:
                players_keypoints_detection = players_keypoints_detection.unsqueeze(0)

            for player_keypoints_detection in players_keypoints_detection:
                player_keypoints = PlayerKeypoints(
                    player_keypoints=[
                        PlayerKeypoint(
                            id=i,
                            name=PlayerKeypoints.KEYPOINTS_NAMES[i],
                            xy=(
                                keypoint[0].item() * ratio_x,
                                keypoint[1].item() * ratio_y,
                            )
                        )
                        for i, keypoint in enumerate(player_keypoints_detection)
                    ]
                )

                players_keypoints.append(player_keypoints)
            
            predictions.append(PlayersKeypoints(players_keypoints))
        
        return predictions
    
    def predict_frames(self, frame_generator: Iterable[np.ndarray], **kwargs):
        raise NoPredictFrames()


    