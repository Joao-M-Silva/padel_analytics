from typing import Literal
import numpy as np
import cv2
from enum import Enum

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
    

def euclidean_distance(v1: tuple[int, int], v2: tuple[int, int]) -> float:
    return np.sqrt(
        (v1[0] - v2[0])**2 + (v1[1] - v2[1])**2
    )

def estimate_velocity(
    frame_index: int,
    fps: int,
    players_detections: list[list[Player]],
    ball_detections: list[Ball],
    keypoints_detections: list[list[Keypoint]],
    frame_interval: int = 3,
    converter: VelocityConverter = VelocityConverter.KM_PER_H,
):
    assert len(players_detections) == len(ball_detections)

    delta_time = frame_interval / fps

    players_detection = players_detections[frame_index]
    players_detection = sorted(
        players_detection,
        key=lambda x: x.id,
    )
    keypoints_detection = keypoints_detections[frame_index]
    keypoints_detection = sorted(
        keypoints_detection, 
        key=lambda x: x.id,
    )
    source_keypoints = [
        list(keypoint.xy)
        for keypoint in keypoints_detection
    ]

    ball_detection_t0 = ball_detections[frame_index]
    ball_detection_t1 = ball_detections[frame_index + frame_interval]

    distances = []
    for player_detection in players_detection: 
        distances.append(
            euclidean_distance(
                tuple(float(x) for x in ball_detection_t0.xy),
                tuple(float(x) for x in player_detection.feet),
            )
        )
    
    player_index = distances.index(min(distances))

    ball_detection_t0 = np.array(ball_detection_t0.xy)
    ball_detection_t1 = np.array(ball_detection_t1.xy)
    r = np.array(ball_detection_t1.xy) - np.array(ball_detection_t0.xy)
    
    ball_position_t0 = np.array(
        [
            ball_detection_t0.xy[0],
            float(players_detection[player_index].feet[1]),
        ]
    )
    translation = ball_position_t0 - ball_detection_t0
    ball_position_t1 = ball_detection_t1 + translation
    r_proj = ball_position_t1 - ball_position_t0

    assert r == r_proj

    src_keypoints = np.array(source_keypoints, np.float32)

    perspective_transformer = PerspectiveTransformer(src_keypoints)

    x0_p, y0_p = perspective_transformer.project_point(ball_position_t0)
    x1_p, y1_p = perspective_transformer.project_point(ball_position_t1)

    Vx = (x1_p - x0_p) / delta_time
    Vy = (y1_p - y0_p) / delta_time

    multiplier = converter.value
    Vx, Vy = Vx*multiplier, Vy*multiplier

    return Vx, Vy