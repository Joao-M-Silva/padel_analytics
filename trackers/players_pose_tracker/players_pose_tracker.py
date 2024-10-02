from typing import Literal, Iterable
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
from ultralytics import YOLO


@dataclass
class PoseKeypoint:

    id: int
    name: str
    xy: tuple[float, float]

    def asint(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.xy)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "xy": self.xy,
        }
    
    @classmethod
    def from_dict(cls, pose_keypoint: dict):
        return cls(**pose_keypoint)
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        x, y = self.xy

        cv2.circle(
            frame,
            (int(x), int(y)),
            radius=2,
            color=(255, 0, 0),
            thickness=-1,
        )

        return frame
    
class PoseKeypoints:

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

    def __init__(self, pose_keypoints: list[PoseKeypoint]):
        if pose_keypoints == []:
            self.keypoints = {}
        else:
            self.keypoints = {
                keypoint.name: keypoint.asint()
                for keypoint in pose_keypoints
            }
        
        self.pose_keypoints = pose_keypoints
            

    def to_dict(self) -> dict:
        return {
            "pose_keypoints": [ 
                keypoint.to_dict()
                for keypoint in self.pose_keypoints
            ]
        }
    
    @classmethod
    def from_dict(cls, pose_keypoints: dict):
        pose_keypoints = [
            PoseKeypoint.from_dict(keypoint)
            for keypoint in pose_keypoints
        ]
        return cls(pose_keypoints)

    def draw(self, frame: np.ndarray) -> np.ndarray:

        if self.keypoints == {}:
            return frame

        frame = frame.copy()

        for connection in self.CONNECTIONS:
            cv2.line(
                frame, 
                self.keypoints[connection[0]],
                self.keypoints[connection[1]],
                color=(255, 0, 0),
                thickness=2,
            )

        return frame

    
class PlayersPoseTracker:

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(
        self, 
        model_path: str, 
        train_image_size: Literal[640, 1280],
    ):
        self.model = YOLO(model_path)

        assert train_image_size in (640, 1280)

        self.train_image_size = train_image_size

    def parse_detections(
        self,
        detections: list[list[PoseKeypoints]],
    ) -> list:
        parsable_detections = []
        for players_pose_detection in detections:
            parsable_players_pose_detection = []
            for player_pose_detection in players_pose_detection:  
                parsable_players_pose_detection.append(
                    [
                        keypoint.to_dict()
                        for keypoint in player_pose_detection.pose_keypoints
                    ]
                )

            parsable_detections.append(parsable_players_pose_detection)
        
        return parsable_detections

    def load_detections(
        self, 
        path: str | Path,
    ) -> list[list[PoseKeypoints]]:
        
        print("Loading Pose Keypoints Detections")

        with open(path, "r") as f:
            parsable_players_pose_detections = json.load(f)
        
        players_pose_detections = []
        for players_pose_detection in parsable_players_pose_detections:
            players_pose_detections.append(
                [
                    PoseKeypoints.from_dict(player_pose_detection)
                    for player_pose_detection in players_pose_detection
                ]
            )
        
        print("Done.")

        return players_pose_detections

    def detect_frame(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.7,
    ) -> list[PoseKeypoints]:
        
        keypoints_names = [
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
        
        h_frame, w_frame = frame.shape[:2] 
        ratio_x = w_frame / self.train_image_size
        ratio_y = h_frame / self.train_image_size

        result = self.model.predict(
            Image.fromarray(frame).resize(
                (self.train_image_size, self.train_image_size),
            ),
            conf=conf,
            iou=iou,
            imgsz=self.train_image_size,
            device=self.DEVICE,
            classes=[0],
        )[0]

        players_keypoints = [] 
        players_pose_keypoints = result.keypoints.xy.squeeze(0)
        if len(players_pose_keypoints.shape) == 2:
            players_pose_keypoints = players_pose_keypoints.unsqueeze(0)

        for player_pose_keypoints in players_pose_keypoints:
            player_keypoints = []
            for i, keypoint in enumerate(player_pose_keypoints):
                player_keypoints.append(
                    PoseKeypoint(
                        id=i,
                        name=keypoints_names[i],
                        xy=(keypoint[0].item() * ratio_x, keypoint[1].item() * ratio_y),
                    )
                )
            
            players_keypoints.append(PoseKeypoints(player_keypoints))

        return players_keypoints
    
    def detect_frames(
        self,
        frames: Iterable[np.ndarray],
        save_path: str | Path = None,
        load_path: str | Path = None,
        conf: float = 0.25,
        iou: float = 0.7,
    ) -> list[list[PoseKeypoints]]:
        
        if load_path is not None:
            players_pose_detections = self.load_detections(load_path)
            
            return players_pose_detections

        self.model.to(self.DEVICE)
        players_keypoints_detections = []
        for frame in frames:
            players_keypoints_detection = self.detect_frame(
                frame,
                conf,
                iou,
            )

            players_keypoints_detections.append(
                players_keypoints_detection
            )

        if save_path is not None and load_path is None:

            print("Saving Pose Keypoints Detections...")

            with open(save_path, "w") as f:
                json.dump(
                    self.parse_detections(players_keypoints_detections),
                    f,
                )
        
        return players_keypoints_detections

    def draw_single_frame(
        self,
        frame: np.ndarray,
        players_keypoints_detection: list[PoseKeypoints],
    ) -> np.ndarray:
        for player_keypoints in players_keypoints_detection:
            frame = player_keypoints.draw(frame)

        return frame
    
    def draw_multiple_frames(
        self, 
        frames: Iterable[np.ndarray], 
        players_keypoints_detections: list[list[PoseKeypoints]],
    ) -> list[np.ndarray]:
        output_frames = []
        for frame, players_keypoints_detection in zip(
            frames,
            players_keypoints_detections,
        ):
            frame = self.draw_single_frame(frame, players_keypoints_detection)
            output_frames.append(frame)

        return output_frames