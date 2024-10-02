from typing import Any, Iterable
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch


class Tracker(ABC):

    @property
    def DEVICE(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def processor(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes a given frame accordingly to the model input requirements
        """
        pass

    def sampler(
        self,
        generator: Iterable[np.ndarray],
        sequence_length: int,
        drop_last: bool = False,
    ) -> Iterable[list[np.ndarray]]:
        """
        Sample sequence_length frames
        """
        w = []
        for x in generator:
            w.append(self.processor(x))

            if len(w) == sequence_length:
                yield w
                w = []

        if not drop_last and w != []:
            yield w

    @abstractmethod
    def load_predictions(self, path: str | Path) -> list[Any]:
        pass

    @abstractmethod
    def predict_frames(self, frames: Iterable[np.ndarray]) -> list[Any]:
        """
        Prediction over a sample of frames
        """
        pass

    @abstractmethod
    def draw_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw frame prediction 
        """
        pass


        



