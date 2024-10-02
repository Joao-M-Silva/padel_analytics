from typing import Any, Iterable
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch


class Tracker(ABC):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def load_detections(self, path: str | Path) -> list[Any]:
        pass

    @abstractmethod
    def detect_frames(self, frames: Iterable[np.ndarray]) -> list[Any]:
        pass
        



