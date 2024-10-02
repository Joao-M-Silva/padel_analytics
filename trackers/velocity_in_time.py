from typing import Iterable, Any
from dataclasses import dataclass
import numpy as np
import cv2
import math

from trackers.ball_tracker.ball_tracker import Ball

@dataclass
class VelocityVector:

    r0: tuple[float, float]
    r1: tuple[float, float]

    @property
    def vector(self) -> tuple[float, float]:
        return (
            self.r1[0] - self.r0[0],
            self.r1[1] - self.r0[1],
        )

    @staticmethod
    def angle_between_vectors(
        u: tuple[float, float], 
        v: tuple[float, float],
    ) -> tuple[float, float]:
        dot_product = sum(i*j for i, j in zip(u, v))
        norm_u = math.sqrt(sum(i**2 for i in u))
        norm_v = math.sqrt(sum(i**2 for i in v))
        cos_theta = dot_product / (norm_u * norm_v)
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def angle(self, vector: "VelocityVector"):
        return VelocityVector.angle_between_vectors(
            self.vector, vector.vector,
        )
    
    def draw_velocity_vector(self, frame: np.ndarray):
        image = frame.copy()
        print(image.shape)
        image = cv2.arrowedLine(
            image, 
            tuple(int(x) for x in self.r0), 
            tuple(int(x) for x in self.r1), 
            (255, 0, 0), 
            6,
        )  
        
        return image


def generator_chuncks(
    sequence: Iterable[Any],
    sequence_length: int,
) -> Iterable[list[Any]]:
    w = []
    for x in sequence:
        w.append(x)
        if len(w) == sequence_length:
            yield list(w)
            del w[0]


def get_velocity_vector_per_frame_interval(
    ball_detections: list[Ball],
    fps: float,
) -> list[VelocityVector]:
    
    delta_time = 1 / fps

    velocity_vectors = []
    for ball_detection_t0, ball_detection_t1 in generator_chuncks(ball_detections, 2):
        velocity_vectors.append(
            VelocityVector(
                ball_detection_t0.xy,
                ball_detection_t1.xy,
            ),
        )
    
    return velocity_vectors

def get_velocity_vectors_angle_per_frame_interval(
    velocity_vectors: list[VelocityVector]
) -> list[float]:
    angles = []
    for vector0, vector1 in generator_chuncks(velocity_vectors, 2):
        try:
            angles.append(vector0.angle(vector1))
        except ZeroDivisionError:
            print("zero division")
            angles.append(0.0)
    
    return angles