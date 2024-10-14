from typing import Iterable, Literal, Type, Optional
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import supervision as sv

from utils import converters
from trackers.tracker import Object, Tracker, NoPredictFrames


class Player:

    """
    Player detection in a given video frame
    
    Attributes:
        detection: player bounding box detection
        projection: player position in a 2D court projection
    """

    def __init__(
        self, 
        detection: sv.Detections, 
        projection: Optional[tuple[int, int]] = None,
    ):
        self.detection = detection
        self.projection = projection
        self.xyxy = detection.xyxy[0]
        self.id = (
            int(detection.tracker_id[0]) 
            if detection.tracker_id 
            else None
        )
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
    def from_json(cls, x: dict):
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

    def serialize(self) -> dict:
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
        """
        Draw player detection in a given frame

        Parameters:
            frame: frame of interest
            video_info: source video information like fps and resolution
            annotator: bounding box style
            show_confidence: True to write detection confidence
        """

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

        box_annotator = annotators[annotator](
            thickness=thickness, 
            color=sv.Color.BLUE,
        )

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
        if self.projection:
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
        else:
            raise ValueError("Inexistent projection.")
    

class Players(Object):

    """
    Players detection in a given video frame
    """

    def __init__(self, players: list[Player]):
        super().__init__()
        self.players = players

    @classmethod
    def from_json(cls, x: list[dict]) -> "Players":
        return cls(
            players=[
                Player.from_json(player_json)
                for player_json in x
            ]
        )

    def serialize(self) -> list[dict]:
        return [
            player.serialize()
            for player in self.players
        ]

    def __len__(self) -> int:
        return len(self.players)

    def __iter__(self) -> Iterable[Player]:
        return (player for player in self.players)
    
    def __getitem__(self, i: int) -> Player:
        return self.players[i]
    
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
        """
        Draw players detection in a given frame

        Parameters:
            frame: frame of interest
            video_info: source video information like fps and resolution
            annotator: bounding box style
            show_confidence: True to write detection confidence
        """
    
        for player in self.players:
            frame = player.draw(
                frame, 
                video_info, 
                annotator, 
                show_confidence,
            )

        return frame


class PlayerTracker(Tracker):

    """
    Tracker of players object

    Attributes:
        model_path: yolo model path
        annotator: bounding box style
        show_confidence: True to write detection confidence 
        load_path: serializable tracker results path 
        save_path: path to save serializable tracker results
    """

    CONF = 0.5
    IOU = 0.7
    IMGSZ = 640

    def __init__(
        self, 
        model_path: str,
        polygon_zone: sv.PolygonZone,
        batch_size: int,
        annotator: Literal[
            "rectangle_bounding_box",
            "round_bounding_box",
            "corner_bounding_box",
            "ellipse"
        ] = "rectangle_bounding_box",
        show_confidence: bool = True,
        load_path: Optional[str | Path] = None,
        save_path: Optional[str | Path] = None,
    ):
        super().__init__(
            load_path=load_path,
            save_path=save_path,
        )

        self.model = YOLO(model_path)
        self.polygon_zone = polygon_zone
        self.batch_size = batch_size
        self.annotator = annotator
        self.show_confidence = show_confidence

    def video_info_post_init(self, video_info: sv.VideoInfo) -> "PlayerTracker":
        self.video_info = video_info
        self.byte_track = sv.ByteTrack(frame_rate=video_info.fps)
        return self

    def object(self) -> Type[Object]:
        return Players

    def draw_kwargs(self) -> dict:
        return {
            "video_info": self.video_info,
            "annotator": self.annotator,
            "show_confidence": self.show_confidence,
        }
    
    def __str__(self) -> str:
        return "players_tracker"
    
    def restart(self) -> None:
        """
        Reset the tracking results
        """
        self.results.restart()
        print(f"{self.__str__()}: Byte tracker reset")
        self.byte_track.reset()

    def processor(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def to(self, device: str) -> None:
        self.model.to(device)

    def predict_sample(self, sample: Iterable[np.ndarray], **kwargs) -> list[Players]:
        """
        Prediction over a sample of frames
        """

        sample = [
            self.processor(frame)
            for frame in sample
        ]

        results = self.model.predict(
            sample, 
            conf=self.CONF,
            iou=self.IOU,
            imgsz=self.IMGSZ,
            device=self.DEVICE,
            # max_det=4,
            classes=[0],
        )

        predictions = []
        for result in results:
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[
                self.polygon_zone.trigger(detections)
            ]
            detections = self.byte_track.update_with_detections(
                detections=detections,
            )

            predictions.append(
                Players(
                    [
                        Player(detection=detections[i])
                        for i in range(len(detections))
                    ]
                )
            )

        return predictions
    
    def predict_frames(self, frame_generator: Iterable[np.ndarray], **kwargs):
        raise NoPredictFrames()
        