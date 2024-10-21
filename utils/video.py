""" Functions to read and save videos """

from typing import Literal
import cv2
import numpy as np
from pathlib import Path

from utils import converters


def read_video(
    path: str | Path, 
    max_frames: int = None,
) -> tuple[list[np.ndarray], int, int, int]:
    
    print("Reading Video ...")

    cap = cv2.VideoCapture(path)
    w, h = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(
            frame, 
            cv2.COLOR_BGR2RGB,
        )

        frames.append(frame_rgb)
        
        if max_frames is not None:
            if len(frames) >= max_frames:
                break

    cap.release()

    print("Done.")

    return frames, fps, w, h

def save_video(
    frames: list[np.ndarray],
    path: str | Path,
    fps: int,
    h: int,
    w: int,
):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    for frame in frames:
        frame_bgr = cv2.cvtColor(
            frame, 
            cv2.COLOR_RGB2BGR,
        )
        out.write(frame_bgr)
    out.release()