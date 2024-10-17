""" Definition of object tracking abstractions """

from typing import Iterable, Optional, Type, Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import supervision as sv


class NoPredictSample(Exception):
    """
    Raise this exception when the tracker predicts based on the
    frame generator and not based on samples
    """
    pass

class NoPredictFrames(Exception):
    """
    Raise this exception when the tracker predicts based on samples
    and not based on a frame generator
    """
    pass


class Object(ABC):

    """
    Abstraction of an object to be tracked

    e.g. players, ball
    """

    @classmethod
    def from_json(cls, x: dict | list[dict]) -> "Object":
        """
        Create an instance of an object from a dictionary

        Parameters:
            x: dictionary with values to use in the initializer
        Returns:
            an object instance
        """
        pass

    def serialize(self) -> dict | list[dict]:
        """
        Return a serializable representation of an object

        Returns:
            a serializable object representation
        """
        pass

    def draw(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """
        Draw object prediction in a given frame
        """
        pass


@dataclass
class TrackingResults:
    
    """
    Tracking results over a sequence of frames

    Attributes:
        predictions: list of each frame tracking results
        sample_predictions: frames sample tracking results
        counter: counts the number of updates
    """

    predictions: list[Object] = field(default_factory=lambda: [])
    sample_predictions: list[Object] = field(default_factory=lambda: [])
    counter: int = 0

    def load(self, predictions: list[Object]) -> None:
        """
        Load video object predictions
        """
        self.predictions = predictions
        self.sample_predictions = []
        self.counter: int = 0

    def update(self, predictions: list[Object]) -> None:
        """
        Add object predictions to the current state

        Parameters:
            predictions: tracking predictions to add
        """
        self.predictions += predictions
        self.sample_predictions = predictions
        self.counter += 1

    def restart(self) -> None:
        """
        Restart the predictions state
        """
        self.predictions = []
        self.sample_predictions = []
        self.counter = 0

    def __len__(self) -> int:
        """
        Retrives the current number of frames with tracking results
        """
        return len(self.predictions)
    
    def __getitem__(self, i: int) -> Object:
        return self.predictions[i]
    
    def __iter__(self) -> Iterable[Object]:
        return (pred for pred in self.predictions)


class Tracker(ABC):

    """
    Abstraction of an object tracker

    Attributes:
        results: tracking results
        load_path: serializable tracker results path 
        save_path: path to save serializable tracker results
    """

    batch_size : int

    def __init__(
        self, 
        load_path: Optional[str | Path] = None,
        save_path: Optional[str | Path] = None,
    ) -> None:
        
        self.results = TrackingResults()
        self.load_path = load_path
        self.save_path = save_path

        # Load predictions if load_path is not None
        self.load_predictions()

    @abstractmethod
    def video_info_post_init(self, video_info: sv.VideoInfo) -> "Tracker":
        """
        Declare attributes dependent of the source video information

        Parameters:
            video_info: source video information like fps and resolution
        """
        pass

    @abstractmethod
    def object(self) -> Type[Object]:
        """
        Retrieves the object subclass 
        """
        pass

    @abstractmethod
    def draw_kwargs(self) -> dict:
        """
        Retrieves a dictionary with tracking object drawing parameters
        """
        pass
    
    @property
    def DEVICE(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def restart(self) -> None:
        """
        Reset the tracking results
        """
        pass

    def __len__(self) -> int:
        """
        Retrieves the number of predictions in the current state
        """
        return len(self.results)

    @abstractmethod
    def __str__(self) -> str:
        """
        Overwrite the string representation magic method in order to
        retrieve a unique tracker string identifier

        Returns:
            a unique tracker identifier
        """
        pass

    def save_predictions(self) -> None:
        """
        Save parsable predictions

        Parameters: 
            path: saved predictions file path
        """

        if self.save_path:

            print(f"{self.__str__()}: Saving predictions ...")
            
            parsable_predictions = [
                object_cls.serialize()
                for object_cls in self.results.predictions
            ]
            
            with open(self.save_path, "w") as f:
                json.dump(parsable_predictions, f)
            
            print(f"{self.__str__()}: {self.__len__()} predictions saved.")
    
    def load_predictions(self) -> None:
        """
        Load previously saved predictions
        """

        if self.load_path:

            print(f"{self.__str__()}: Loading predictions ...")

            with open(self.load_path, "r") as f:
                parsable_detections = json.load(f)
            
            predictions = [
                self.object().from_json(obj_json)
                for obj_json in parsable_detections
            ]

            self.results.load(predictions)
        
        print(f"{self.__str__()}: {self.__len__()} predictions loaded.")

    def to(self, device: Literal["cuda", "cpu"]) -> None:
        """
        Move tracker model/models to the given device

        Parameters:
            device: either cpu or gpu
        """
        pass

    @abstractmethod
    def predict_sample(self, sample: Iterable[np.ndarray], **kwargs) -> Optional[list[Object]]:
        """
        Prediction over a sample of frames

        Parameters:
            sample: sample of processed frames
        Returns:
            sample frames object detections
        Raises:
            NoPredictSample when the tracker predicts based on the frame generator
        """
        pass

    @abstractmethod
    def predict_frames(self, frame_generator: Iterable[np.ndarray], **kwargs) -> Optional[list[Object]]:
        """
        Prediction over a video frame generator

        Parameters:
            sample: sample of processed frames
        Returns:
            sample frames object detections
        Raises:
            NoPredictFrames when the tracker predicts based on samples
        """
        pass

    def predict_and_update(self, frame_generator: Iterable[np.ndarray], **kwargs) -> list[Object]:
        """
        Prediction over a video updating the results

        Parameters:
            sample: sample of processed frames
        Returns:
            sample frames object detections
        """

        def sampler(
            generator: Iterable[np.ndarray],
            sequence_length: int,
        ) -> Iterable[list[np.ndarray]]:
            """
            Sample sequence_length frames 

            Parameters:
                generator: frame generator
                sequence_length: number of frames to be retrived
                drop_last: True to drop the last sequence if its incomplete
            Returns:
                a sample of frames
            """
            w = []
            for x in generator:
                w.append(x)

                if len(w) == sequence_length:
                    yield w
                    w = []

            if w != []:
                yield w
        
        try:
            predictions = self.predict_frames(frame_generator, **kwargs)
            self.results.predictions = predictions
        except NoPredictFrames:
            for sample in tqdm(
                sampler(
                    frame_generator,
                    sequence_length=self.batch_size,
                )
            ):
                predictions = self.predict_sample(sample, **kwargs)
                self.results.update(predictions)

        print(f"{self.__str__()}: {len(self.results)} predictions.")
            
        return self.results






    


        



