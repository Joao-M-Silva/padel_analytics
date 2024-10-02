from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
import cv2
from enum import Enum, auto
import pims

from trackers import Player, Ball, Keypoint
from utils.conversions import convert_pixel_distance_to_meters


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
    Vz: Optional[float]

    @property
    def norm(self):
        if self.Vz is None:
            return np.sqrt(self.Vx**2 + self.Vy**2)
        else:
            return np.sqrt(self.Vx**2 + self.Vy**2 + self.Vz**2)

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
        get_Vz: bool = False,
    ) -> Tuple[BallVelocityData, BallVelocity]:
        
        delta_time = (frame_index_t1 - frame_index_t0) / self.fps

        players_detection_t0 = self.players_detections[frame_index_t0]
        players_detection_t0 = sorted(
            players_detection_t0,
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
        
        player_index_t0 = self.find_index_closest_player(
            reference=tuple(float(x) for x in ball_detection_t0.xy),
            players_detection=players_detection_t0,
        )

        ball_position_t0 = np.array(
            [
                ball_detection_t0.xy[0],
                float(players_detection_t0[player_index_t0].feet[1]),
            ]
        )

        ball_detection_t0 = np.array(ball_detection_t0.xy)
        ball_detection_t1 = np.array(ball_detection_t1.xy)

        if impact_type == ImpactType.FLOOR:
            ball_position_t1 = ball_detection_t1
        elif impact_type == ImpactType.RACKET:
            players_detection_t1 = self.players_detections[frame_index_t1]
            players_detection_t1 = sorted(
                players_detection_t1,
                key=lambda x: x.id,
            )
            player_index_t1 = self.find_index_closest_player(
                reference=tuple(float(x) for x in ball_detection_t1),
                players_detection=players_detection_t1,
            )

            ball_position_t1 = np.array(
                [
                    ball_detection_t1[0],
                    float(players_detection_t1[player_index_t1].feet[1]),
                ]
            )

        if get_Vz:
            player_height_pixels_t0 = float(players_detection_t0[player_index_t0].height)

            ball_height_pixels_t0 = abs(
                ball_detection_t0[1]
                -
                ball_position_t0[1]
            )
            ball_height_meters_t0 = convert_pixel_distance_to_meters(
                ball_height_pixels_t0,
                reference_in_meters=1.8,
                reference_in_pixels=player_height_pixels_t0,
            )

            if impact_type == ImpactType.FLOOR:
                Vz = ball_height_meters_t0 / delta_time
            elif impact_type == ImpactType.RACKET:
                player_height_pixels_t1 = float(players_detection_t1[player_index_t1].height)

                ball_height_pixels_t1 = abs(
                    ball_detection_t1[1]
                    -
                    ball_position_t1[1]
                )
                ball_height_meters_t1 = convert_pixel_distance_to_meters(
                    ball_height_pixels_t1,
                    reference_in_meters=1.8,
                    reference_in_pixels=player_height_pixels_t1,
                )
                Vz = (ball_height_meters_t1 - ball_height_meters_t0) / delta_time
        else:
            Vz = None
        
        src_keypoints = np.array(source_keypoints, np.float32)

        perspective_transformer = PerspectiveTransformer(src_keypoints)

        x0_p, y0_p = perspective_transformer.project_point(ball_position_t0)
        x1_p, y1_p = perspective_transformer.project_point(ball_position_t1)

        ball_velocity_data = BallVelocityData(
            frame_index_t0=frame_index_t0,
            frame_index_t1=frame_index_t1,
            player_index=player_index_t0,
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
        print("HERE")
        print(Vz)
        if Vz is not None:
            Vz = Vz*multiplier

        ball_velocity = BallVelocity(converter, Vx, Vy, Vz)

        return ball_velocity_data, ball_velocity