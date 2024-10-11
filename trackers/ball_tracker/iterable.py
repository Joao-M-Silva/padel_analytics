from typing import Literal, Iterable
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import IterableDataset


class BallTrajectoryIterable(IterableDataset):
    def __init__(
        self,
        seq_len: int = 8,
        sliding_step: int = 1,
        data_mode: Literal["heatmap", "coordinate"] = "heatmap",
        bg_mode: Literal["", "subtract", "subtract_concat", "concat"] = "",
        frame_alpha: int = -1,
        frame_generator: Iterable[np.ndarray] = None,
        pred_dict: dict = None,
        HEIGHT: int = 288,
        WIDTH: int = 512,
        SIGMA: float = 2.5,
        IMG_FORMAT: str = "png",
        median: np.ndarray = None,
        median_range: int = 300,
        padding: bool = False
    ):
        assert data_mode in ['heatmap', 'coordinate']
        assert bg_mode in ['', 'subtract', 'subtract_concat', 'concat']

        super(BallTrajectoryIterable).__init__()

        self.frame_generator = frame_generator
        self.pred_dict = pred_dict
        self.padding = padding and self.sliding_step == self.seq_len

        # Image size
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.IMG_FORMAT = IMG_FORMAT

        # Gaussian heatmap parameters
        self.mag = 1
        self.sigma = SIGMA
        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.data_mode = data_mode
        self.bg_mode = bg_mode
        self.frame_alpha = frame_alpha

        self.frames_in_memory = []

        if self.frame_generator is not None:

            # For TrackNet inference
            assert self.data_mode == 'heatmap'

            self.data_dict = None
            self.img_config = None

            if self.bg_mode:
                if median is None:
                    print("Calculating median ...")
                    print("1. Getting frames")
                    for frame in self.frame_generator:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.frames_in_memory.append(frame)
                        if len(self.frames_in_memory) == median_range:
                            break
                    
                    print("2. Calculating")
                    median = np.median(
                        np.array(self.frames_in_memory), 
                        0,
                    )
                    print("Done.")

                if self.bg_mode == 'concat':
                    median = Image.fromarray(
                        median.astype("uint8")
                    )
                    median = np.array(median.resize(size=(self.WIDTH, self.HEIGHT)))
                    self.median = np.moveaxis(median, -1, 0)
                else:
                    self.median = median

        elif self.pred_dict is not None:
            
            # For InpaintNet inference
            assert self.data_mode == 'coordinate'

            self.data_dict, self.img_config = self._gen_input_from_pred_dict()

    
    def _gen_input_from_pred_dict(self) -> tuple[dict, dict]:

        """ Generate input sequences from a prediction dictionary. """

        id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
        coor_pred = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
        pred_vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
        inpaint_mask = np.array([], dtype=np.float32).reshape(0, self.seq_len)
        x_pred, y_pred, vis_pred = self.pred_dict['x'], self.pred_dict['y'], self.pred_dict['visibility']
        inpaint = self.pred_dict['inpaint_mask']
        assert len(x_pred) == len(y_pred) == len(vis_pred) == len(inpaint), \
            f'Length of x_pred, y_pred, vis_pred and inpaint are not equal.'
        
        # Sliding on the frame sequence
        last_idx = -1
        for i in range(0, len(inpaint), self.sliding_step):
            tmp_idx, tmp_coor_pred, tmp_vis_pred, tmp_inpaint = [], [], [], []
            # Construct a single input sequence
            for f in range(self.seq_len):
                if i+f < len(inpaint):
                    tmp_idx.append((0, i+f))
                    tmp_coor_pred.append((x_pred[i+f], y_pred[i+f]))
                    tmp_vis_pred.append(vis_pred[i+f])
                    tmp_inpaint.append(inpaint[i+f])
                    last_idx = i+f
                else:
                    # Padding the last sequence if imcompleted
                    if self.padding:
                        tmp_idx.append((0, last_idx))
                        tmp_coor_pred.append((x_pred[last_idx], y_pred[last_idx]))
                        tmp_vis_pred.append(vis_pred[last_idx])
                        tmp_inpaint.append(inpaint[last_idx])
                    else:
                        break
                
            if len(tmp_idx) == self.seq_len:
                assert len(tmp_coor_pred) == len(tmp_inpaint), \
                    f'Length of predicted coordinates and inpaint masks are not equal.'
                id = np.concatenate((id, [tmp_idx]), axis=0)
                coor_pred = np.concatenate((coor_pred, [tmp_coor_pred]), axis=0)
                pred_vis = np.concatenate((pred_vis, [tmp_vis_pred]), axis=0)
                inpaint_mask = np.concatenate((inpaint_mask, [tmp_inpaint]), axis=0)
        
        return (
            dict(
                id=id, 
                coor_pred=coor_pred, 
                pred_vis=pred_vis, 
                inpaint_mask=inpaint_mask,
            ),
            dict(
                img_scaler=self.pred_dict['img_scaler'], 
                img_shape=self.pred_dict['img_shape'],
            ),
        )


    def median_image(self) -> Image:
        return Image.fromarray(np.moveaxis(self.median, 0, -1))

    def generator_chuncks(
        self, 
        generator: Iterable[np.ndarray],
        sequence_length: int,
    ) -> Iterable[np.array]:
        w = []
        for x in generator:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            w.append(x)

            if len(w) == sequence_length:
                yield np.array(w)
                del w[0]

    def process_chunck(self, imgs: np.array) -> np.array:

        if self.bg_mode:
            median_img = self.median
            
        # Process the frame sequence
        frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
        for i in range(self.seq_len):
            img = Image.fromarray(imgs[i])
            if self.bg_mode == 'subtract':
                img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
                img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                img = img.reshape(1, self.HEIGHT, self.WIDTH)
            elif self.bg_mode == 'subtract_concat':
                diff_img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
                diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                img = np.moveaxis(img, -1, 0)
                img = np.concatenate((img, diff_img), axis=0)
            else:
                img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                img = np.moveaxis(img, -1, 0)
                
            frames = np.concatenate((frames, img), axis=0)
            
        if self.bg_mode == 'concat':
            frames = np.concatenate((median_img, frames), axis=0)
            
        # Normalization
        frames /= 255.

        return frames

    def __iter__(self) -> Iterable[np.array]:

        if self.data_mode != 'heatmap':
            raise Exception("Not yet implemented")

        if self.frames_in_memory:
            for frame_chunck in self.generator_chuncks(
                self.frames_in_memory, self.seq_len,
            ):
                frames = self.process_chunck(frame_chunck)
                yield frames

        for frame_chunck in self.generator_chuncks(
            self.frame_generator, self.seq_len,
        ):
            frames = self.process_chunck(frame_chunck)
            yield frames