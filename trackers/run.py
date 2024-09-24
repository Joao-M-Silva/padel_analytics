from typing import Iterable, Union
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import cv2

from trackers.players_tracker.players_tracker import PlayerTracker, Player
from trackers.ball_tracker.ball_tracker import BallTracker


class PlayerTrackerRunner:

    def __init__(
        self,
        player_tracker: PlayerTracker,
        save_path: str | Path,
        load_path: str | Path,
    ):
        self.player_tracker = player_tracker
        self.save_path = save_path
        self.load_path = load_path

    def run(self, frame: np.ndarray) -> list[Player]:
        self.player_tracker.model.to(self.player_tracker.DEVICE)
        return self.player_tracker.detect_frame(frame)

class BallTrackerRunner:
    pass


@dataclass
class Trackers:
    
    player_tracker: PlayerTrackerRunner
    ball_tracker: BallTrackerRunner


def run_trackers(
    trackers: Trackers,
    frames: Iterable[np.ndarray],
):
    
    if trackers.player_tracker.load_path is not None:
        player_detections = trackers.player_tracker.player_tracker.load_detections(
            trackers.player_tracker.load_path,
        )
        run_player_detections = False
    else:
        run_player_detections = True
        player_detections = []

    if trackers.ball_tracker.load_path is not None:
        ball_detections = trackers.ball_tracker.ball_tracker.load_detections(
            trackers.ball_tracker.load_path,
        )
        run_ball_detections = False
    else:
        run_ball_detections = True
        ball_detections = []

    for i, frame in enumerate(frames):
        frame = cv2.cvtColor(
            frame, 
            cv2.COLOR_BGR2RGB,
        )
        if run_player_detections:
            players = trackers.player_tracker.run(frame)
            player_detections.append(players)

        

    
    if trackers.player_tracker.save_path is not None:
        print("Saving Players Detections...")

        with open(trackers.player_tracker.save_path, "w") as f:
            json.dump(
                trackers.player_tracker.player_tracker.parse_detections(player_detections),
                f,
            )

        print("Done.")

    if trackers.ball_tracker.save_path is not None:
        print("Saving Ball Detections...")

        parsable_ball_detections = [
            ball_detection.to_dict()
            for ball_detection in ball_detections
        ]

        with open(trackers.ball_tracker.save_path, "w") as f:
            json.dump(
                parsable_ball_detections,
                f,
            )
            
        print("Done.")
    
    


