from typing import Tuple
from dataclasses import dataclass
import numpy as np
import cv2
from enum import Enum, auto
import pims

from trackers import Player, Ball, Keypoint


class VelocityConverter(Enum):

    M_PER_S = 1
    FT_PER_S = 3.28
    KM_PER_H = 3.6
    MILERS_PER_H = 2.24


class PerspectiveTransformer:

    PADEL_COURT_KEYPOINTS = np.array(
        [
            [-5, 10],
            [5, 10],
            [-5, 7],
            [0, 7],
            [5, 7],
            [-5, 0],
            [5, 0],
            [-5, -7],
            [0, -7],
            [5, -7],
            [-5, -10],
            [5, -10],
        ],
        np.float32,
    )

    def __init__(self, src_keypoints: np.ndarray):
        src_keypoints = src_keypoints.astype(np.float32)
        self.H, _ = cv2.findHomography(
            src_keypoints,
            self.PADEL_COURT_KEYPOINTS,
        )
    
    def project_point(self, point: np.ndarray):
        src_point = point.astype(np.float32)
        src_point = np.append(
            src_point,
            np.array([1]),
            axis=0,
        )
        dst_point = np.matmul(self.H, src_point)
        dst_point = dst_point / dst_point[2]
        return (dst_point[0], dst_point[1])
    

class ImpactType(Enum):

    FLOOR = auto()
    RACKET = auto()

@dataclass
class BallVelocityData:

    frame_index_t0: int
    frame_index_t1: int    
    player_index: int
    detection_t0: np.ndarray
    detection_t1: np.ndarray
    position_t0: np.ndarray
    position_t1: np.ndarray
    position_t0_proj: np.ndarray
    position_t1_proj: np.ndarray

    def draw_velocity(self, video: pims.pyav_reader.PyAVReaderTimed):
        image = np.array(video[self.frame_index_t0]).copy()
        print(image.shape)
        for point0, point1, color in zip(
            [self.detection_t0, self.position_t0],
            [self.detection_t1, self.position_t1],
            [(0, 0, 255), (0, 255, 0)]
        ):
            image = cv2.arrowedLine(
                image, 
                tuple(int(x) for x in point0), 
                tuple(int(x) for x in point1), 
                color, 
                6,
            )  
        
        return image

        
@dataclass
class BallVelocity:

    converter: VelocityConverter
    Vx: float
    Vy: float

    @property
    def norm(self):
        return np.sqrt(self.Vx**2 + self.Vy**2)

class BallVelocityEstimator:

    def __init__(
        self,
        source_video_fps: int,
        players_detections: list[list[Player]],
        ball_detections: list[Ball],
        keypoints_detections: list[list[Keypoint]],
    ):
        self.fps = source_video_fps
        self.players_detections = players_detections
        self.ball_detections = ball_detections
        self.keypoints_detections = keypoints_detections

        assert len(players_detections) == len(ball_detections)

    @staticmethod
    def euclidean_distance(
        v1: tuple[int, int], 
        v2: tuple[int, int],
    ) -> float:
        return np.sqrt(
            (v1[0] - v2[0])**2 + (v1[1] - v2[1])**2
        )
    
    def find_index_closest_player(
        self, 
        reference: Tuple[float, float],
        players_detection: list[Player],
    ) -> int:
        distances = []
        for player_detection in players_detection: 
            distances.append(
                self.euclidean_distance(
                    reference,
                    tuple(float(x) for x in player_detection.feet),
                )
            )
        
        return distances.index(min(distances))
    
    def estimate_velocity(
        self,
        frame_index_t0: int,
        frame_index_t1: int,
        impact_type: ImpactType,
        converter: VelocityConverter = VelocityConverter.KM_PER_H,
    ) -> Tuple[BallVelocityData, BallVelocity]:
        
        delta_time = (frame_index_t1 - frame_index_t0) / self.fps

        players_detection = self.players_detections[frame_index_t0]
        players_detection = sorted(
            players_detection,
            key=lambda x: x.id,
        )

        keypoints_detection = self.keypoints_detections[frame_index_t0]
        keypoints_detection = sorted(
            keypoints_detection, 
            key=lambda x: x.id,
        )

        source_keypoints = [
            list(keypoint.xy)
            for keypoint in keypoints_detection
        ]

        ball_detection_t0 = self.ball_detections[frame_index_t0]
        ball_detection_t1 = self.ball_detections[frame_index_t1]

        print(ball_detection_t0)
        print(ball_detection_t1)
        
        player_index = self.find_index_closest_player(
            reference=tuple(float(x) for x in ball_detection_t0.xy),
            players_detection=players_detection,
        )

        ball_position_t0 = np.array(
            [
                ball_detection_t0.xy[0],
                float(players_detection[player_index].feet[1]),
            ]
        )

        ball_detection_t0 = np.array(ball_detection_t0.xy)
        ball_detection_t1 = np.array(ball_detection_t1.xy)

        if impact_type == ImpactType.FLOOR:
            ball_position_t1 = ball_detection_t1
        elif impact_type == ImpactType.RACKET:
            players_detection = self.players_detections[frame_index_t1]
            players_detection = sorted(
                players_detection,
                key=lambda x: x.id,
            )
            player_index = self.find_index_closest_player(
                reference=tuple(float(x) for x in ball_detection_t1),
                players_detection=players_detection,
            )

            ball_position_t1 = np.array(
                [
                    ball_detection_t1[0],
                    float(players_detection[player_index].feet[1]),
                ]
            )
        
        src_keypoints = np.array(source_keypoints, np.float32)

        perspective_transformer = PerspectiveTransformer(src_keypoints)

        x0_p, y0_p = perspective_transformer.project_point(ball_position_t0)
        x1_p, y1_p = perspective_transformer.project_point(ball_position_t1)

        ball_velocity_data = BallVelocityData(
            frame_index_t0=frame_index_t0,
            frame_index_t1=frame_index_t1,
            player_index=player_index,
            detection_t0=ball_detection_t0,
            detection_t1=ball_detection_t1,
            position_t0=ball_position_t0,
            position_t1=ball_position_t1,
            position_t0_proj=np.array([x0_p, y0_p]),
            position_t1_proj=np.array([x1_p, y1_p]),
        )

        Vx = (x1_p - x0_p) / delta_time
        Vy = (y1_p - y0_p) / delta_time

        multiplier = converter.value
        Vx, Vy = Vx*multiplier, Vy*multiplier

        ball_velocity = BallVelocity(converter, Vx, Vy)

        return ball_velocity_data, ball_velocity