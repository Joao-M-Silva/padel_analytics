""" Functions to convert between image datatypes """

import numpy as np
from PIL import Image
import base64
import io

def numpy_to_pil(image_array: np.ndarray) -> Image:
    return Image.fromarray(image_array.astype("uint8"))

def numpy_to_base64(image_array: np.ndarray) -> str:
    image_pil = numpy_to_pil(image_array)
    stream = io.BytesIO()
    image_pil.save(stream, format="PNG")
    bytes_obj = stream.getvalue()
    image_b64 = base64.b64encode(bytes_obj).decode("utf-8")

    return image_b64

def pil_to_numpy(image_pil: Image) -> np.ndarray:
    return np.asarray(image_pil)

def base64_to_pil(image_b64: str) -> Image:
    decoded = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(decoded))

def base64_to_numpy(image_b64: str) -> np.ndarray:
    image_pil = base64_to_pil(image_b64)
    return pil_to_numpy(image_pil)