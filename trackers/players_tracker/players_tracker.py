from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from utils import converters


@dataclass
class Player:

    """
    Definition of a player

    Attributes:
        id: player unique identifier
        xyxy: player bounding box coordinates
        projection: player position mini court projection 
    """

    id: int
    xyxy: tuple[float, float, float, float]
    projection: tuple[int, int] = None

    @property
    def top_left(self) -> tuple[int, int]:
        return tuple(
            int(p)
            for p in self.xyxy[:2]
        )
    
    @property
    def bottom_right(self) -> tuple[int, int]:
        return tuple(
            int(p)
            for p in self.xyxy[2:]
        )
    
    @property
    def height(self) -> float:
        return self.bottom_right[1] - self.top_left[1]
    
    @property
    def width(self) -> float:
        return self.bottom_right[0] - self.top_left[0]
    
    @property
    def midpoint(self) -> tuple[int, int]:
        return (
            int(self.top_left[0] + self.width / 2),
            int(self.top_left[1] + self.height / 2),
        )
    
    @property
    def feet(self) -> tuple[int, int]:
        return (
            int(self.top_left[0] + self.width / 2),
            int(self.bottom_right[1]),
        )
    
    @classmethod
    def from_dict(cls, x: dict):
        return cls(**x)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "xyxy": self.xyxy,
        }

    def draw(self, frame: np.ndarray) -> np.ndarray:
        text_location = (
            self.top_left[0],
            self.top_left[1] - 10,
        )
        cv2.putText(
            frame, 
            str(self.id),
            text_location,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )
        cv2.rectangle(
            frame,
            self.top_left,
            self.bottom_right,
            (0, 0, 255),
            2,
        )

        return frame
    
    def draw_projection(self, frame: np.ndarray) -> np.ndarray:
        cv2.circle(
            frame,
            self.projection,
            8,
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            frame, 
            str(self.id),
            (
                self.projection[0], 
                self.projection[1] - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

        return frame


class PlayerTracker:

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect_frame(self, frame: np.ndarray) -> list[Player]:
        result = self.model.track(
            converters.numpy_to_pil(frame), 
            persist=True,
            conf=0.5,
            iou=0.7,
            imgsz=640,
            device=self.DEVICE,
            max_det=4,
            classes=[0],
        )[0]

        players = []
        for box in result.boxes:
            try:
                track_id = int(box.id.item())
            except AttributeError:
                track_id = 0

            xyxy = box.xyxy.tolist()[0]

            assert box.cls.item() == 0

            players.append(
                Player(id=track_id, xyxy=xyxy),
            )

        return players
    
    def parse_detections(
        self, 
        detections: list[list[Player]],
    ) -> list:
        parsable_detections = []
        for detection in detections:
            parsable_detections.append(
                [
                    player.to_dict()
                    for player in detection
                ]
            )

        return parsable_detections
    
    def load_detections(
        self, 
        path: str | Path,
    ) -> list[list[Player]]:
        
        print("Loading Players Detections ...")

        with open(path, "r") as f:
            parsable_player_detections = json.load(f)

        players_detections = []
        for player_detection in parsable_player_detections:
            players_detections.append(
                [
                    Player.from_dict(player)
                    for player in player_detection
                ]
            )
        
        print("Done.")

        return players_detections
        
    def detect_frames(
        self, 
        frames: list[np.ndarray],
        save_path: str | Path = None,
        load_path: str | Path = None,
    ) -> list[list[Player]]:
        
        if load_path is not None:
            player_detections = self.load_detections(load_path)
            
            return player_detections
        
        self.model.to(self.DEVICE)
        
        # YOLO models don't have batch prediction
        player_detections = []
        for frame in frames:
            players = self.detect_frame(frame)
            player_detections.append(players)

        if save_path is not None and load_path is None:

            print("Saving Players Detections...")

            with open(save_path, "w") as f:
                json.dump(
                    self.parse_detections(player_detections),
                    f,
                )

            print("Done.")

        self.model.to("cpu")
        
        return player_detections
    
    def draw_single_frame(
        self, 
        frame: np.ndarray, 
        players_detection: list[Player],
    ) -> np.ndarray:
        for player in players_detection:
            frame = player.draw(frame)

        return frame
    
    def draw_multiple_frames(
        self, 
        frames: list[np.ndarray], 
        players_detections: list[list[Player]],
    ) -> list[np.ndarray]:
        output_frames = []
        for frame, players_detection in zip(frames, players_detections):
            frame = self.draw_single_frame(frame, players_detection)
            output_frames.append(frame)

        return output_frames