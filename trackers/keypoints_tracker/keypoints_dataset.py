from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from utils import converters


class InferenceKeypointDataset(Dataset):

    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames
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

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index: int):

        image_pil = converters.numpy_to_pil(self.frames[index])
        image = self.transforms(image_pil)

        return {
            "image": image, 
            "array": np.array(image_pil.resize((self.w, self.h))),
        }
    