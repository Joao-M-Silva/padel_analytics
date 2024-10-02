from typing import Iterable, Literal
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import supervision as sv

from utils import converters
from trackers.tracker import Tracker


class Player:

    """
    Definition of a player

    Note - projection: player position mini court projection 
    """

    def __init__(
        self, 
        detection: sv.Detections, 
        projection: tuple[int, int] = None,
    ):
        self.detection = detection
        self.projection = projection
        self.xyxy = detection.xyxy[0]
        self.id = int(detection.tracker_id[0]) if detection.tracker_id else None
        self.class_id = int(detection.class_id[0])
        self.confidence = float(detection.confidence[0])
       
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
        try:
            projection = x["projection"]
        except KeyError:
            projection = None
            
        detection = sv.Detections(
            xyxy=np.array([x["xyxy"]]),
            confidence=np.array([x["confidence"]]),
            tracker_id=np.array([x["id"]]),
            class_id=np.array([x["class_id"]]),
        )
        return cls(detection=detection, projection=projection)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "xyxy": [float(p) for p in self.xyxy],
            "projection": self.projection,
            "class_id": self.class_id,
            "confidence": self.confidence,
        }

    def draw(
        self, 
        frame: np.ndarray, 
        video_info: sv.VideoInfo,
        annotator: Literal[
            "rectangle_bounding_box",
            "round_bounding_box",
            "corner_bounding_box",
            "ellipse"
        ] = "rectangle_bounding_box",
        show_confidence: bool = True,
    ) -> np.ndarray:

        thickness = sv.calculate_optimal_line_thickness(
            resolution_wh=video_info.resolution_wh,
        )
        text_scale = sv.calculate_optimal_text_scale(
            resolution_wh=video_info.resolution_wh,
        )
        annotators = {
            "rectangle_bounding_box": sv.BoxAnnotator,
            "round_bounding_box": sv.RoundBoxAnnotator,
            "corner_bounding_box": sv.BoxCornerAnnotator,
            "ellipse": sv.EllipseAnnotator,
        }

        box_annotator = annotators[annotator](thickness=thickness, color=sv.Color.BLUE)
        label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.TOP_CENTER,
            text_scale=text_scale,
            text_thickness=thickness,
            color=sv.Color.BLUE,
        )

        annotated_frame = cv2.cvtColor(
            frame, 
            cv2.COLOR_RGB2BGR,
        ).copy()

        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=self.detection,
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=self.detection,
            labels=[
                f"{self.id}: {self.confidence:.2f}" 
                if show_confidence 
                else f"{self.id}"
            ]
        )

        return cv2.cvtColor(
            annotated_frame, 
            cv2.COLOR_BGR2RGB,
        )
    
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


class PlayerTracker(Tracker):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(
        self, 
        model_path: str,
        video_info: sv.VideoInfo,
        polygon_zone: sv.PolygonZone,
    ):
        self.model = YOLO(model_path)
        self.video_info = video_info
        self.byte_track = sv.ByteTrack(frame_rate=video_info.fps)
        self.polygon_zone = polygon_zone

    def detect_frame(self, frame: np.ndarray) -> list[Player]:
        result = self.model.predict(
            frame, 
            conf=0.5,
            iou=0.7,
            imgsz=640,
            device=self.DEVICE,
            # max_det=4,
            classes=[0],
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[
            self.polygon_zone.trigger(detections)
        ]
        detections = self.byte_track.update_with_detections(detections=detections)
        
        players = []
        for i in range(len(detections)):
            detection = detections[i]
            players.append(Player(detection=detections[i]))

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
        frames: Iterable[np.ndarray],
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
            frame = cv2.cvtColor(
                frame, 
                cv2.COLOR_BGR2RGB,
            )
            players = self.detect_frame(frame)
            player_detections.append(players)

        if save_path is not None and load_path is None:

            print("Saving Players Detections...")

            parsed_detections = self.parse_detections(player_detections)
            with open(save_path, "w") as f:
                json.dump(
                    parsed_detections,
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
            frame = player.draw(frame, self.video_info)

        return frame
    
    def draw_multiple_frames(
        self, 
        frames: Iterable[np.ndarray], 
        players_detections: list[list[Player]],
    ) -> list[np.ndarray]:
        output_frames = []
        for frame, players_detection in zip(frames, players_detections):
            frame = self.draw_single_frame(frame, players_detection)
            output_frames.append(frame)

        return output_frames