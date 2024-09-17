from typing import Literal
from dataclasses import dataclass
import cv2
import numpy as np

from constants import BASE_LINE, SIDE_LINE, SERVICE_SIDE_LINE, NET_SIDE_LINE
from utils import convert_meters_to_pixel_distance, convert_pixel_distance_to_meters
from trackers import Player, Keypoint, Ball


class InconsistentPredictedKeypoints(Exception):
    pass


PointPixels = tuple[int, int]

#def euclidean_distance(point1: PointPixels, point2: PointPixels) -> float:
#    return np.sqrt(
#        (point1[0] - point2[0])**2
#        +
#        (point1[1] - point2[1])**2
#    )

@dataclass
class Rectangle:

    top_left: PointPixels
    bottom_right: PointPixels

    @property
    def width(self) -> int:
        return self.bottom_right[0] - self.top_left[0]
    
    @property
    def height(self) -> int:
        return self.bottom_right[1] - self.top_left[1]
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def perimeter(self) -> int:
        return 2*self.width + 2*self.height

@dataclass
class CourtKeypoints:

    """
    Court 12 points of interest to be detected 
    """

    k1: PointPixels
    k2: PointPixels
    k3: PointPixels
    k4: PointPixels
    k5: PointPixels
    k6: PointPixels
    k7: PointPixels
    k8: PointPixels
    k9: PointPixels
    k10: PointPixels
    k11: PointPixels
    k12: PointPixels

    def __post_init__(self):
        # Calculate reference origin
        self.origin = self._get_origin()

    @property
    def width(self) -> int:
        return self.k7[0] - self.k6[0]
    
    @property 
    def height(self) -> int:
        return self.k1[1] - self.k11[1]

    def _get_origin(self) -> PointPixels:
        """
        Get the reference origin to estimate positions in meters
        """
        k6origin = (
            int((self.k7[0] - self.k6[0]) / 2), 
            int((self.k7[1] - self.k6[1]) / 2),
        )
        origin = (
            self.k6[0] + k6origin[0], 
            self.k6[1] + k6origin[1],
        )
        return origin

    @property
    def keypoints(self) -> list[Keypoint]:
        return [
            Keypoint(id=i, xy=tuple(float(p) for p in v))
            for i, v in enumerate(self.__dict__.values())
        ]

    def lines(self) -> list[tuple[PointPixels, PointPixels]]:
        """
        Court lines start and end keypoints
        """
        return [
            (self.k1, self.k2),
            (self.k3, self.k5),
            (self.k6, self.k7),
            (self.k8, self.k10),
            (self.k11, self.k12),
            (self.k1, self.k11),
            (self.k4, self.k9),
            (self.k2, self.k12),
        ]
    
    def shift_point_origin(
        self, 
        point: PointPixels,
        dimension: Literal["pixels", "meters"],
    ) -> PointPixels:
        """
        Change the origin of the point to the middle of the 
        mini court. If dimension is meters the vector entries
        are converted from pixels to meters
        """
        shifted_point = tuple(
            point[0] - self.origin[0],
            point[1] - self.origin[1],
        )
        if dimension == "meters":
            shifted_point[0] = convert_pixel_distance_to_meters(
                pixel_distance=shifted_point[0],
                reference_in_meters=BASE_LINE,
                reference_in_pixels=self.width,
            )
            shifted_point[1] = convert_pixel_distance_to_meters(
                pixel_distance=shifted_point[1],
                reference_in_meters=BASE_LINE,
                reference_in_pixels=self.width,
            )

        return shifted_point
    

    
    
class MiniCourt:

    def __init__(self, frame: np.ndarray):
        h, w, _ = frame.shape
        # Canvas background points in pixels
        self.WIDTH = int(0.14*w)
        self.HEIGHT = int(0.47*h)
        self.BUFFER = 50
        self.PADDING = 20

        self._set_canvas_background_position(frame)
        self._set_mini_court_position()
        self._set_mini_court_keypoints()

        # Initialize the homography matrix H
        self.H = None

    def _set_canvas_background_position(self, frame: np.ndarray) -> None:
        
        """
        Set the canvas background position
        """
        
        end_x = frame.shape[1] - self.BUFFER
        end_y = self.BUFFER + self.HEIGHT
        start_x = end_x - self.WIDTH
        start_y = end_y - self.HEIGHT

        self.background_position = Rectangle(
            top_left=(int(start_x), int(start_y)),
            bottom_right=(int(end_x), int(end_y))
        )

    def _set_mini_court_position(self) -> None:

        """
        Set the mini court position (respecting measurements in meters) 
        inside the canvas background
        """

        court_start_x = self.background_position.top_left[0] + self.PADDING
        court_start_y = self.background_position.top_left[1] + self.PADDING
        court_end_x = self.background_position.bottom_right[0] - self.PADDING
        court_width = court_end_x - court_start_x
        court_height = convert_meters_to_pixel_distance(
            SIDE_LINE,
            reference_in_meters=BASE_LINE,
            reference_in_pixels=court_width
        )
        court_end_y = court_start_y + court_height

        self.court_position = Rectangle(
            top_left=(int(court_start_x), int(court_start_y)),
            bottom_right=(int(court_end_x), int(court_end_y)),
        )
        
    def _set_mini_court_keypoints(self) -> None:

        """
        Set the mini court 12 points of interest
        """

        service_line_height = convert_meters_to_pixel_distance(
            SERVICE_SIDE_LINE,
            reference_in_meters=BASE_LINE,
            reference_in_pixels=self.court_position.width,
        )

        self.court_keypoints = CourtKeypoints(
            k1=(
                self.court_position.top_left[0],
                self.court_position.bottom_right[1],
            ),
            k2=self.court_position.bottom_right,
            k3=(
                self.court_position.top_left[0],
                self.court_position.bottom_right[1] - service_line_height,
            ),
            k4=(
                int(self.court_position.top_left[0] + self.court_position.width / 2),
                self.court_position.bottom_right[1] - service_line_height,
            ),
            k5=(
                self.court_position.bottom_right[0],
                self.court_position.bottom_right[1] - service_line_height,
            ),
            k6=(
                self.court_position.top_left[0],
                int(self.court_position.top_left[1] + self.court_position.height / 2),
            ),
            k7=(
                self.court_position.bottom_right[0],
                int(self.court_position.top_left[1] + self.court_position.height / 2),
            ),
            k8=(
                self.court_position.top_left[0],
                self.court_position.top_left[1] + service_line_height,
            ),
            k9=(
                int(self.court_position.top_left[0] + self.court_position.width / 2),
                self.court_position.top_left[1] + service_line_height,
            ),
            k10=(
                self.court_position.bottom_right[0],
                self.court_position.top_left[1] + service_line_height,
            ),
            k11=self.court_position.top_left,
            k12=(
                self.court_position.bottom_right[0],
                self.court_position.top_left[1],
            )
        )

    def draw_background_single_frame(
        self, 
        frame: np.ndarray, 
        alpha: float = 0.5,
    ) -> np.ndarray:

        """
        Draw the canvas background on the given frame
        """

        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(
            shapes,
            self.background_position.top_left,
            self.background_position.bottom_right,
            (255, 255, 255),
            -1,
        )
        output_frame = frame.copy()
        mask = shapes.astype(bool)
        output_frame[mask] = cv2.addWeighted(
            output_frame,
            alpha,
            shapes,
            1 - alpha,
            0,
        )[mask]

        # output_frame = cvw. cvtColor(output_frame, cv2.BGR2RGB)

        return output_frame
    
    def draw_mini_court_single_frame(self, frame: np.ndarray) -> np.ndarray:

        """
        Draw minicourt points of interest and lines 
        """

        output_frame = frame.copy()
        for k, v in self.court_keypoints.__dict__.items():
            if "k" in k:
                cv2.circle(
                    output_frame,
                    v,
                    5,
                    (255, 0, 0),
                    -1,
                )
            else:
                cv2.circle(
                    output_frame,
                    v,
                    5,
                    (0, 255, 0),
                    -1,
                )
                

        for line in self.court_keypoints.lines():
            start_point = line[0]
            end_point = line[1]
            cv2.line(
                output_frame,
                start_point,
                end_point,
                (0, 0, 0),
                2,
            )
        
        return output_frame
    
    def homography_matrix(self, keypoints_detection: list[Keypoint]) -> np.ndarray:

        """
        Calculates the homography matrix that projects the court keypoints detected
        on a given frame into the mini court

        Parameters:
            keypoints_detection: predicted keypoints on a single frame
        """

        keypoints_detection = sorted(keypoints_detection, key=lambda x: x.id)
        src_points = np.array(
            [
                keypoint.xy
                for keypoint in keypoints_detection
            ]
        ) # keypoints of the given frame
        dst_points = np.array(
            [
                keypoint.xy
                for keypoint in self.court_keypoints.keypoints
            ]
        ) # keypoints on the mini court

        if src_points.shape != dst_points.shape:
            raise InconsistentPredictedKeypoints("Don't have enough source points")

        H, _ = cv2.findHomography(src_points, dst_points)

        return H
    
    def project_point(
        self,
        point: tuple[int, int],
        homography_matrix: np.ndarray = None,
        keypoints_detection: list[Keypoint] = None,
    ) -> tuple[float, float]:
        
        """
        Project point given a homography matrix H.
        If the homography matrix is not provided than one needs to provide the 
        keypoints detected on a given frame to calculate the respective 
        homography matrix.
        
        Parameters:
            point: point to be projected
            homography_matrix: homography matrix that projects into the mini court plane
            keypoints_detection: court keypoints detected on a given frame

        Returns:
            projected point
        """

        assert homography_matrix.shape == (3, 3)

        if homography_matrix is None and keypoints_detection is None:
            raise ValueError("Not enough data to make projection into mini court")
        
        if homography_matrix is None:
            homography_matrix = self.homography_matrix(keypoints_detection)

        src_point = np.array([float(p) for p in point])
        src_point = np.append(
            src_point,
            np.array([1]),
            axis=0,
        )

        dst_point = np.matmul(homography_matrix, src_point)
        dst_point = dst_point / dst_point[2]

        return (dst_point[0], dst_point[1])
    
    def project_player(
        self, 
        player_detection: Player,
        keypoints_detection: list[Keypoint] = None,
        homography_matrix: np.ndarray = None,
    ) -> Player:
        
        """
        Mini court projection of a player detection
        """
        
        projected_point = self.project_point(
            point=player_detection.feet,
            homography_matrix=homography_matrix,
            keypoints_detection=keypoints_detection,
        )

        player_detection.projection = tuple(int(v) for v in projected_point)

        return player_detection 
    
    def project_ball(
        self, 
        ball_detection: Ball,
        keypoints_detection: list[Keypoint] = None,
        homography_matrix: np.ndarray = None,
    ) -> Ball:
        
        """
        Mini court projection of a ball detection
        """
        
        projected_point = self.project_point(
            point=ball_detection.asint(),
            homography_matrix=homography_matrix,
            keypoints_detection=keypoints_detection,
        )

        ball_detection.projection = tuple(int(v) for v in projected_point)

        return ball_detection

    def draw_projected_player(
        self, 
        frame: np.ndarray,
        player_detection: Player,
        homography_matrix: np.ndarray = None,
        keypoints_detection: list[Keypoint] = None,
    ) -> np.ndarray:
        output_frame = frame.copy()
        projected_player = self.project_player(
            player_detection=player_detection,
            homography_matrix=homography_matrix,
            keypoints_detection=keypoints_detection,    
        )

        output_frame = projected_player.draw_projection(output_frame)
        
        return output_frame
    
    def draw_projected_players(
        self,
        frame: np.ndarray,
        players_detection: list[Player],
        homography_matrix: np.ndarray = None,
        keypoints_detection: list[Keypoint] = None,   
    ) -> np.ndarray:
        output_frame = frame.copy()
        for player_detection in players_detection:
            output_frame = self.draw_projected_player(
                frame=output_frame,
                player_detection=player_detection,
                keypoints_detection=keypoints_detection,
                homography_matrix=homography_matrix
            )
        
        return output_frame
    
    def draw_projected_ball(
        self,
        frame: np.ndarray,
        ball_detection: Ball,
        homography_matrix: np.ndarray = None,
        keypoints_detection: list[Keypoint] = None,
    ) -> np.ndarray:
        output_frame = frame.copy()
        projected_ball = self.project_ball(
            ball_detection=ball_detection,
            homography_matrix=homography_matrix,
            keypoints_detection=keypoints_detection,
        )

        output_frame = projected_ball.draw_projection(output_frame)

        return output_frame

    def draw_minicourt_with_projections(
        self, 
        frames: list[np.ndarray],
        keypoints_detections: list[list[Keypoint]],
        players_detections: list[list[Player]],
        ball_detections: list[Ball],
        data_analytics: DataAnalytics = None,
    ):
        
        if collect_data:
            data = {
                "frame": [],
                "ball_position": [],
                "player_1_position": [],
            }

        # SUBOPTIMAL
        homography_matrix = self.homography_matrix(keypoints_detections[0])
        
        output_frames = []
        for i, frame in enumerate(frames):
            output_frame = self.draw_background_single_frame(frame.copy())
            output_frame = self.draw_mini_court_single_frame(output_frame)
            output_frame = self.draw_projected_players(
                output_frame, 
                players_detection=players_detections[i],
                homography_matrix=homography_matrix,
            )
            output_frame = self.draw_projected_ball(
                output_frame,
                ball_detection=ball_detections[i],
                homography_matrix=homography_matrix,
            )
            output_frames.append(output_frame)

        return output_frames
    
    

    

    
