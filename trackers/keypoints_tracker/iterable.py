from typing import Iterable
import numpy as np
import cv2
from torch.utils.data import IterableDataset
from torchvision import transforms

from utils import converters


class KeypointsIterable(IterableDataset):

    def __init__(self, frame_generator: Iterable[np.ndarray]):
        self.frame_generator = frame_generator
        self.h = 224
        self.w = 224
        self.transforms = transforms.Compose(
            [
                transforms.Resize((self.h, self.w)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.465, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ]
        )
    
    def __iter__(self) -> Iterable[np.ndarray]:

        for frame in self.frame_generator:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = converters.numpy_to_pil(frame)
            image = self.transforms(image_pil)

            self.h_frame, self.w_frame = frame.shape[:2]

            yield {
                "image": image, 
                "array": np.array(image_pil.resize((self.w, self.h))),
            }

       
    