"""
Ball velocity estimation from two video frames.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import math

import cv2
import numpy as np

# Virtual court dimensions in pixels used for homography computation.
# Width = 500px  →  10 m (BASE_LINE)
# Height = 1000px →  20 m (SIDE_LINE)
_COURT_WIDTH_PX: int = 500
_COURT_HEIGHT_PX: int = 1000
_SCALE: float = _COURT_WIDTH_PX / 10.0  # 50 px / m


def _build_virtual_court_points() -> np.ndarray:
    """
    Return the 12 destination keypoints for the virtual flat court (pixels).

    Layout (matches ProjectedCourtKeypoints in analytics/projected_court.py):

        k11(0,0)----------------k12(W,0)
        |                            |
        k8(0,srv)----k9(W/2,srv)---k10(W,srv)
        |                            |
        k6(0,H/2)-------------------k7(W,H/2)
        |                            |
        k3(0,H-srv)--k4(W/2,H-srv)--k5(W,H-srv)
        |                            |
        k1(0,H)------------------k2(W,H)
    """
    W = _COURT_WIDTH_PX
    H = _COURT_HEIGHT_PX
    srv = int(3 / 20 * H)  # SERVICE_SIDE_LINE / SIDE_LINE * H = 150 px

    points = [
        (0, H),        # k1
        (W, H),        # k2
        (0, H - srv),  # k3
        (W // 2, H - srv),  # k4
        (W, H - srv),  # k5
        (0, H // 2),   # k6
        (W, H // 2),   # k7
        (0, srv),      # k8
        (W // 2, srv), # k9
        (W, srv),      # k10
        (0, 0),        # k11
        (W, 0),        # k12
    ]
    return np.array(points, dtype=np.float32)


_VIRTUAL_COURT_POINTS = _build_virtual_court_points()


class ImpactType(Enum):
    FLOOR = "floor"
    RACKET = "racket"


@dataclass
class BallVelocityData:
    """Projected positions and raw pixel positions for velocity visualisation."""

    position_t0_proj: tuple  # (x, y) in metres, origin at court centre
    position_t1_proj: tuple  # (x, y) in metres, origin at court centre
    frame_t0: int
    frame_t1: int
    ball_xy_t0: tuple        # pixel coords in source video at t0
    ball_xy_t1: tuple        # pixel coords in source video at t1

    def draw_velocity(self, video) -> np.ndarray:
        """
        Draw a velocity arrow on the t0 frame.

        Parameters:
            video: pims.Video (or any object supporting index access returning
                   array-like frames)

        Returns:
            numpy.ndarray frame with arrow drawn
        """
        frame = np.array(video[self.frame_t0])
        pt0 = tuple(int(v) for v in self.ball_xy_t0)
        pt1 = tuple(int(v) for v in self.ball_xy_t1)
        if pt0 != (0, 0) and pt1 != (0, 0):
            frame = cv2.arrowedLine(frame, pt0, pt1, (0, 255, 255), 4, tipLength=0.3)
            cv2.circle(frame, pt0, 8, (0, 255, 0), -1)
            cv2.circle(frame, pt1, 8, (0, 0, 255), -1)
        return frame


@dataclass
class BallVelocity:
    """Ball speed components (m/s)."""

    norm: float         # resultant speed in m/s
    vx: float = 0.0    # x component (m/s)
    vy: float = 0.0    # y component (m/s)
    vz: float = 0.0    # vertical component estimate (m/s)

    def __str__(self) -> str:
        return (
            f"BallVelocity("
            f"norm={self.norm:.2f} m/s / {self.norm * 3.6:.1f} km/h, "
            f"vx={self.vx:.2f}, vy={self.vy:.2f}, vz={self.vz:.2f})"
        )


class BallVelocityEstimator:
    """
    Estimates ball velocity between two video frames using court homography.

    Parameters:
        source_video_fps: frames per second of the source video
        players_detections: per-frame player detection results (unused directly
            but kept for API compatibility)
        ball_detections: list of Ball objects indexed by frame number
        keypoints_detections: list of Keypoints objects indexed by frame number
    """

    def __init__(
        self,
        source_video_fps: float,
        players_detections,
        ball_detections: list,
        keypoints_detections: list,
    ) -> None:
        self.fps = source_video_fps
        self.ball_detections = ball_detections
        self.keypoints_detections = keypoints_detections

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_homography(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Find the best homography matrix near *frame_index*.

        Searches up to 30 frames in either direction for a frame that has
        exactly 12 valid court keypoints.
        """
        total = len(self.keypoints_detections)
        search_range = min(31, total)
        for offset in range(search_range):
            for idx in (frame_index + offset, frame_index - offset):
                if 0 <= idx < total:
                    kp = self.keypoints_detections[idx]
                    if kp is not None and hasattr(kp, "keypoints") and len(kp.keypoints) == 12:
                        src_pts = np.array(
                            [k.xy for k in kp.keypoints], dtype=np.float32
                        )
                        H, _ = cv2.findHomography(src_pts, _VIRTUAL_COURT_POINTS)
                        if H is not None:
                            return H
        return None

    def _project_point(self, point: tuple, H: np.ndarray) -> tuple:
        """Project a pixel point to virtual court coordinates."""
        pt = np.array([float(point[0]), float(point[1]), 1.0])
        proj = H @ pt
        proj /= proj[2]
        return (float(proj[0]), float(proj[1]))

    def _to_meters_centered(self, court_px: tuple) -> tuple:
        """
        Convert virtual court pixel coords to metres with origin at court centre.

        Virtual court origin (0, 0) is top-left; centre is (250, 500).
        """
        cx = _COURT_WIDTH_PX / 2.0
        cy = _COURT_HEIGHT_PX / 2.0
        x_m = (court_px[0] - cx) / _SCALE
        y_m = (court_px[1] - cy) / _SCALE
        return (x_m, y_m)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_velocity(
        self,
        frame_t0: int,
        frame_t1: int,
        impact_type: ImpactType,
        get_Vz: bool = False,
    ) -> tuple:
        """
        Estimate ball velocity between *frame_t0* and *frame_t1*.

        Parameters:
            frame_t0: index of the first frame
            frame_t1: index of the second frame (must be > frame_t0)
            impact_type: ImpactType.FLOOR or ImpactType.RACKET
            get_Vz: if True, add a rough vertical component estimate

        Returns:
            (BallVelocityData, BallVelocity)
        """
        if frame_t1 <= frame_t0:
            raise ValueError("frame_t1 must be greater than frame_t0")

        delta_time = (frame_t1 - frame_t0) / self.fps

        # Retrieve ball detections
        n = len(self.ball_detections)
        if frame_t0 >= n or frame_t1 >= n:
            raise ValueError(
                f"Frame index out of range (have {n} detections, "
                f"requested {frame_t0} and {frame_t1})"
            )
        ball_t0 = self.ball_detections[frame_t0]
        ball_t1 = self.ball_detections[frame_t1]

        xy_t0 = ball_t0.xy if ball_t0.visibility else (0.0, 0.0)
        xy_t1 = ball_t1.xy if ball_t1.visibility else (0.0, 0.0)

        # Compute homography
        H = self._get_homography(frame_t0)
        if H is None:
            raise ValueError(
                "Could not compute homography: no valid court keypoints found "
                "near the requested frame."
            )

        # Project ball positions to virtual court, then convert to metres
        proj_t0_px = self._project_point(xy_t0, H)
        proj_t1_px = self._project_point(xy_t1, H)

        proj_t0_m = self._to_meters_centered(proj_t0_px)
        proj_t1_m = self._to_meters_centered(proj_t1_px)

        # Velocity components
        vx = (proj_t1_m[0] - proj_t0_m[0]) / delta_time
        vy = (proj_t1_m[1] - proj_t0_m[1]) / delta_time

        vz = 0.0
        if get_Vz and impact_type == ImpactType.FLOOR:
            # Rough kinematic estimate: ball leaves floor at ~45° after bounce
            d2d = math.sqrt(vx ** 2 + vy ** 2)
            vz = d2d * 0.5

        speed = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

        velocity_data = BallVelocityData(
            position_t0_proj=proj_t0_m,
            position_t1_proj=proj_t1_m,
            frame_t0=frame_t0,
            frame_t1=frame_t1,
            ball_xy_t0=xy_t0,
            ball_xy_t1=xy_t1,
        )
        velocity = BallVelocity(norm=speed, vx=vx, vy=vy, vz=vz)

        return velocity_data, velocity
