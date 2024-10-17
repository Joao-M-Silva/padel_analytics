from typing import Literal, Iterable, Optional, Type
from collections import deque
import json
from dataclasses import dataclass
from pathlib import Path
import math
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, IterableDataset
import torch
import supervision as sv

from trackers.ball_tracker.models import TrackNet, InpaintNet
from trackers.ball_tracker.dataset import BallTrajectoryDataset
from trackers.ball_tracker.iterable import BallTrajectoryIterable
from trackers.ball_tracker.predict import predict, predict_modified
from trackers.tracker import Object, Tracker, NoPredictSample



def get_model(
    model_name: Literal["TrackNet", "InpaintNet"], 
    seq_len: int = None, 
    bg_mode: Literal["", "subtract", "subtract_concat", "concat"] = None,
) -> torch.nn.Module:
    """ 
    Create model by name and the configuration parameter.

    Parameters:
        model_name: type of model to create
            Choices:
                - 'TrackNet': Return TrackNet model
                - 'InpaintNet': Return InpaintNet model
        
        seq_len: length of TrackNet input sequence 
        bg_mode: background mode of TrackNet
            Choices:
                - '': return TrackNet with L x 3 input channels (RGB)
                - 'subtract': return TrackNet with L x 1 input channel 
                    (Difference frame)
                - 'subtract_concat': return TrackNet with L x 4 input channels
                    (RGB + Difference frame)
                - 'concat': return TrackNet with (L+1) x 3 input channels (RGB)

    Returns:
        model with specified configuration
    """

    if model_name == 'TrackNet':
        if bg_mode == 'subtract':
            model = TrackNet(in_dim=seq_len, out_dim=seq_len)
        elif bg_mode == 'subtract_concat':
            model = TrackNet(in_dim=seq_len*4, out_dim=seq_len)
        elif bg_mode == 'concat':
            model = TrackNet(in_dim=(seq_len+1)*3, out_dim=seq_len)
        else:
            model = TrackNet(in_dim=seq_len*3, out_dim=seq_len)
    elif model_name == 'InpaintNet':
        model = InpaintNet()
    else:
        raise ValueError('Invalid model name.')
    
    return model


def get_ensemble_weight(
    seq_len: int, 
    eval_mode: Literal["average", "weight"],
) -> torch.Tensor:
    """ 
    Get weight for temporal ensemble.

    Parameters:
        seq_len: Length of input sequence
        eval_mode: Mode of temporal ensemble
            Choices:
                - 'average': return uniform weight
                - 'weight': return positional weight
        
        Returns:
            weight for temporal ensemble
    """

    if eval_mode == 'average':
        weight = torch.ones(seq_len) / seq_len
    elif eval_mode == 'weight':
        weight = torch.ones(seq_len)
        for i in range(math.ceil(seq_len/2)):
            weight[i] = (i+1)
            weight[seq_len-i-1] = (i+1)
        weight = weight / weight.sum()
    else:
        raise ValueError('Invalid mode')
    
    return weight


def generate_inpaint_mask(pred_dict: dict, th_h: float=30) -> list:
    """ 
    Generate inpaint mask form predicted trajectory.

    Parameters:
        pred_dict: prediction result
            Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
        th_h: height threshold (pixels) for y coordinate
        
    Returns:
        inpaint mask
    """
    y = np.array(pred_dict['y'])
    vis_pred = np.array(pred_dict['visibility'])
    inpaint_mask = np.zeros_like(y)
    i = 0 # index that ball start to disappear
    j = 0 # index that ball start to appear
    threshold = th_h
    while j < len(vis_pred):
        while i < len(vis_pred)-1 and vis_pred[i] == 1:
            i += 1
        j = i
        while j < len(vis_pred)-1 and vis_pred[j] == 0:
            j += 1
        if j == i:
            break
        elif i == 0 and y[j] > threshold:
            # start from the first frame that ball disappear
            inpaint_mask[:j] = 1
        elif (i > 1 and y[i-1] > threshold) and (j < len(vis_pred) and y[j] > threshold):
            inpaint_mask[i:j] = 1
        else:
            # ball is out of the field of camera view 
            pass
        i = j
    
    return inpaint_mask.tolist()


class Ball(Object):

    """
    Ball detection in a given video frame

    Attributes:
        frame: frame associated with the given ball detection
        xy: ball position coordinates
        visibility: 1 if the ball is visible in the given frame
        projection: ball position 2d court projection 
    """

    def __init__(
        self, 
        frame: int, 
        xy: tuple[float, float], 
        visibility: Literal[0,1],
        projection: Optional[tuple[int, int]] = None                    
    ):
        super().__init__()

        self.frame = frame
        self.xy = xy
        self.visibility = visibility
        self.projection = projection

    @classmethod
    def from_json(cls, x: dict):
        return cls(**x)

    def serialize(self) -> dict:
        return {
            "frame": self.frame,
            "xy": self.xy,
            "visibility": self.visibility,
            "projection": self.projection,
        }
    
    def asint(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.xy)
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw ball detection in a given frame
        """

        cv2.circle(
            frame,
            self.asint(),
            6,
            (0, 255, 0),
            -1,
        )

        return frame
    
    def draw_projection(self, frame: np.ndarray) -> np.ndarray:
        
        cv2.circle(
            frame,
            self.projection,
            6,
            (255, 255, 0),
            -1,
        )

        return frame


class BallTracker(Tracker):

    """
    Tracker of ball object

    Attributes:
        tracking_model_path: tracknet model path
        inpainting_model_path: inpainting model path
        median_max_sample_num: maximum number of frames to sample for 
            generating median image
        median: background estimation
        load_path: serializable tracker results path 
        save_path: path to save serializable tracker results

    Note: 
        its important to filter frames of interest before feeding the 
        video to the model
    """

    EVAL_MODE: str = "weight"
    TRAJECTORY_LENGTH: int = 8
    
    HEIGHT: int = 288
    WIDTH: int = 512
    SIGMA: float = 2.5
    IMG_FORMAT = 'png'
    
    def __init__(
        self, 
        tracking_model_path: str,
        inpainting_model_path: str,
        batch_size: int,
        median_max_sample_num: int = 1800, 
        median: Optional[np.ndarray] = None,
        load_path: Optional[str | Path] = None,
        save_path: Optional[str | Path] = None,
    ):
        super().__init__(
            load_path=load_path,
            save_path=save_path,
        )

        self.DELTA_T: float = 1 / math.sqrt(self.HEIGHT**2 + self.WIDTH**2)
        self.COOR_TH = self.DELTA_T * 50

        tracknet_ckpt = torch.load(tracking_model_path)
        self.tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']

        assert self.tracknet_seq_len == self.TRAJECTORY_LENGTH

        self.bg_mode = tracknet_ckpt['param_dict']['bg_mode']

        self.tracknet = get_model(
            "TrackNet", 
            self.tracknet_seq_len,
            self.bg_mode,
        )
        self.tracknet.load_state_dict(tracknet_ckpt['model'])
        self.tracknet.eval()

        if inpainting_model_path:
            inpaintnet_ckpt = torch.load(inpainting_model_path)
            self.inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
            self.inpaintnet = get_model('InpaintNet')
            self.inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
        else:
            self.inpaintnet = None

        self.batch_size = batch_size
        self.median_max_sample_num = median_max_sample_num
        self.median = median
    
    def video_info_post_init(self, video_info: sv.VideoInfo) -> "BallTracker":
        self.video_info = video_info
        return self
    
    def object(self) -> Type[Object]:
        return Ball
    
    def draw_kwargs(self) -> dict:
        return {}
    
    def __str__(self) -> str:
        return "ball_tracker"
    
    def restart(self) -> None:
        self.results.restart()

    def processor(self, frame: np.ndarray):
        pass
    
    def draw_traj(self, img, traj, radius=3, color='red') -> np.ndarray:
        """ Draw trajectory on the image.

            Args:
                img (numpy.ndarray): Image with shape (H, W, C)
                traj (deque): Trajectory to draw

            Returns:
                img (numpy.ndarray): Image with trajectory drawn
        """
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
        img = Image.fromarray(img)
        
        for i in range(len(traj)):
            if traj[i] is not None:
                draw_x = traj[i][0]
                draw_y = traj[i][1]
                bbox =  (draw_x - radius, draw_y - radius, draw_x + radius, draw_y + radius)
                draw = ImageDraw.Draw(img)
                draw.ellipse(bbox, fill='rgb(255,255,255)', outline=color)
                del draw
        # img =  cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        return np.array(img)
    
    def draw_multiple_frames(
        self,
        frames: list[np.ndarray],
        ball_detections: list[Ball],
        traj_len=8
    ):

        pred_queue = deque()
        
        output_frames = []
        for frame, ball_detection in zip(frames, ball_detections):
        
            # Check capacity of queue
            if len(pred_queue) >= traj_len:
                pred_queue.pop()
        
            pred_queue.appendleft(
                list(ball_detection.xy)
            ) if ball_detection.visibility else pred_queue.appendleft(None)

            # Draw prediction trajectory
            output_frames.append(self.draw_traj(frame, pred_queue, color='yellow'))

        return output_frames
    
    def modify_pred_dict(self, pred_dict: dict):

        mapping = {
            "X": "x",
            "Y": "y",
            "Visibility": "visibility",
            "Inpaint_Mask": "inpaint_mask",
            "Img_scaler": "img_scaler",
            "Img_shape": "img_shape",
        }

        return {
            k: pred_dict[v]
            for k, v in mapping.items()
        }
    
    def to(self, device: str) -> None:
        self.tracknet.to(device)
        if self.inpaintnet is not None:
            self.inpaintnet.to(device)

    def predict_sample(self, sample: Iterable[np.ndarray], **kwargs):
        raise NoPredictSample()

    def predict_frames(
        self,
        frame_generator: Iterable[np.ndarray],
        total_frames: int,
    ) -> list[Ball]:

        w_scaler, h_scaler = (
            self.video_info.width / self.WIDTH, 
            self.video_info.height / self.HEIGHT,
        )

        img_scaler = (w_scaler, h_scaler)

        tracknet_pred_dict = {
            'frame':[], 
            'x':[], 
            'y':[], 
            'visibility':[], 
            'inpaint_mask': [],
            'img_scaler': img_scaler, 
            'img_shape': (self.video_info.width, self.video_info.height),
        }

        seq_len = self.tracknet_seq_len

        iterable = BallTrajectoryIterable(
            seq_len=seq_len,
            sliding_step=1,
            data_mode="heatmap",
            bg_mode="concat",
            frame_generator=frame_generator,
            HEIGHT=self.HEIGHT,
            WIDTH=self.WIDTH,
            SIGMA=2.5,
            IMG_FORMAT="png",
            median=self.median,
            median_range=self.median_max_sample_num,
        )

        data_loader = DataLoader(
            iterable,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        video_len = total_frames

        ### Init prediction buffer params ###
        # Number of samples of seq_len frames
        num_sample, sample_count = video_len - seq_len + 1, 0
        buffer_size = seq_len - 1
        sample_indices = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
        frame_indices = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
        y_pred_buffer = torch.zeros(
            (
                buffer_size, 
                seq_len, 
                self.HEIGHT, 
                self.WIDTH
            ), 
            dtype=torch.float32,
        )
        # Weights for the frame prediction ensemble along the distinct samples position
        weight = get_ensemble_weight(seq_len, self.EVAL_MODE)

        for x in tqdm(data_loader):
            x = x.float().to(self.DEVICE)

            batch_size = x.shape[0]
            assert seq_len*3 + 3 == x.shape[1] 

            with torch.no_grad():
                y_pred = self.tracknet(x).detach().cpu()
            
            # Concatenate predictions onto the previous predictions buffer
            y_pred_buffer = torch.cat(
                (y_pred_buffer, y_pred), 
                dim=0,
            )

            ensemble_y_pred = torch.empty(
                (0, 1, self.HEIGHT, self.WIDTH), 
                dtype=torch.float32,
            )

            for sample_i in range(batch_size):
                if sample_count < buffer_size:
                    # Incomplete buffer. A given sample first frame have 
                    # not appeared in all frame positions before
                    y_pred = y_pred_buffer[
                        sample_indices + sample_i,
                        frame_indices,
                    ].sum(0) / (sample_count + 1)
                else:
                    # General complete buffer. A given sample first frame
                    # have appeared in all frame positions before
                    y_pred = (
                        y_pred_buffer[
                            sample_indices + sample_i,
                            frame_indices
                        ] * weight[:, None, None]
                    ).sum(0)

                ensemble_y_pred = torch.cat(
                    (
                        ensemble_y_pred, 
                        y_pred.reshape(1, 1, self.HEIGHT, self.WIDTH),
                    ),
                    dim=0,
                )
                sample_count += 1

                if sample_count == num_sample:
                    # The sample above was the last sample
                    y_zero_pad = torch.zeros(
                        (buffer_size, seq_len, self.HEIGHT, self.WIDTH),
                        dtype=torch.float32,
                    )
                    y_pred_buffer = torch.cat(
                        (y_pred_buffer, y_zero_pad),
                        dim=0,
                    )
                    print(seq_len)
                    for frame_i in range(1, seq_len):
                        y_pred = y_pred_buffer[
                            sample_indices + sample_i + frame_i,
                            frame_indices
                        ].sum(0) / (seq_len - frame_i)

                        ensemble_y_pred = torch.cat(
                            (
                                ensemble_y_pred, 
                                y_pred.reshape(1, 1, self.HEIGHT, self.WIDTH),
                            ),
                            dim=0,
                        )

            # Predict
            tmp_pred = predict_modified(
                y_pred=ensemble_y_pred, # first frame prediction of batch_size samples
                img_scaler=img_scaler,
                WIDTH=self.WIDTH,
                HEIGHT=self.HEIGHT,
            )

            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])

            # Update buffer, keep last predictions for ensemble in next iteration
            y_pred_buffer = y_pred_buffer[-buffer_size:]

        if self.inpaintnet is not None:
            self.inpaintnet.eval()
            seq_len = self.inpaintnet_seq_len
            tracknet_pred_dict["inpaint_mask"] = generate_inpaint_mask(
                tracknet_pred_dict, th_h=self.video_info.height*0.05,
            )
            inpaint_pred_dict = {
                'Frame':[], 
                'X':[], 
                'Y':[], 
                'Visibility':[],
            }

            # Create dataset with overlap sampling for temporal ensemble
            dataset = BallTrajectoryDataset(
                seq_len=seq_len, 
                sliding_step=1, 
                data_mode='coordinate', 
                pred_dict=self.modify_pred_dict(tracknet_pred_dict),
                HEIGHT=self.HEIGHT,
                WIDTH=self.WIDTH,
                SIGMA=self.SIGMA,
                IMG_FORMAT=self.IMG_FORMAT,
            )
            data_loader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=False,
            ) # num_workers=num_workers, 

            weight = get_ensemble_weight(seq_len, self.EVAL_MODE)

            # Init buffer params
            num_sample, sample_count = len(dataset), 0
            buffer_size = seq_len - 1
            sample_indices = torch.arange(seq_len) 
            frame_indices = torch.arange(seq_len-1, -1, -1) 
            coor_inpaint_buffer = torch.zeros(
                (buffer_size, seq_len, 2), 
                dtype=torch.float32,
            )

            for (i, coor_pred, inpaint_mask) in tqdm(data_loader):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                batch_size = i.shape[0]
                with torch.no_grad():
                    coor_inpaint = self.inpaintnet(
                        coor_pred.cuda(), 
                        inpaint_mask.cuda(),
                    ).detach().cpu()
                    
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask)
                
                # Thresholding
                th_mask = (
                    (
                        (coor_inpaint[:, :, 0] < self.COOR_TH) 
                        &
                        (coor_inpaint[:, :, 1] < self.COOR_TH)
                    )
                )
                coor_inpaint[th_mask] = 0.

                coor_inpaint_buffer = torch.cat(
                    (coor_inpaint_buffer, coor_inpaint),
                    dim=0,
                )
                ensemble_i = torch.empty(
                    (0, 1, 2), 
                    dtype=torch.float32,
                )
                ensemble_coor_inpaint = torch.empty(
                    (0, 1, 2), 
                    dtype=torch.float32,
                )
                
                for sample_i in range(batch_size):
                    if sample_count < buffer_size:
                        # Imcomplete buffer
                        coor_inpaint = coor_inpaint_buffer[
                            sample_indices + sample_i, 
                            frame_indices,
                        ].sum(0)
                        coor_inpaint /= (sample_count+1)
                    else:
                        # General case
                        coor_inpaint = (
                            coor_inpaint_buffer[
                                sample_indices + sample_i, 
                                frame_indices,
                            ] * weight[:, None]
                        ).sum(0)
                    
                    ensemble_i = torch.cat(
                        (ensemble_i, i[sample_i][0].view(1, 1, 2)), 
                        dim=0,
                    )
                    ensemble_coor_inpaint = torch.cat(
                        (ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), 
                        dim=0,
                    )
                    sample_count += 1

                    if sample_count == num_sample:
                        # Last input sequence
                        coor_zero_pad = torch.zeros(
                            (buffer_size, seq_len, 2), 
                            dtype=torch.float32,
                        )
                        coor_inpaint_buffer = torch.cat(
                            (coor_inpaint_buffer, coor_zero_pad), 
                            dim=0,
                        )
                        
                        for frame_i in range(1, seq_len):
                            coor_inpaint = coor_inpaint_buffer[
                                sample_indices + sample_i + frame_i, 
                                frame_indices
                            ].sum(0)
                            coor_inpaint /= (seq_len - frame_i)
                            ensemble_i = torch.cat(
                                (ensemble_i, i[-1][frame_i].view(1, 1, 2)), 
                                dim=0,
                            )
                            ensemble_coor_inpaint = torch.cat(
                                (ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), 
                                dim=0,
                            )

                # Thresholding
                th_mask = ((ensemble_coor_inpaint[:, :, 0] < self.COOR_TH) & (ensemble_coor_inpaint[:, :, 1] < self.COOR_TH))
                ensemble_coor_inpaint[th_mask] = 0.

                # Predict
                tmp_pred = predict(
                    ensemble_i, 
                    c_pred=ensemble_coor_inpaint,
                    img_scaler=img_scaler, 
                    WIDTH=self.WIDTH, 
                    HEIGHT=self.HEIGHT,
                )

                {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
                # Update buffer, keep last predictions for ensemble in next iteration
                coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]

        pred_dict = inpaint_pred_dict if self.inpaintnet is not None else tracknet_pred_dict
        
        ball_detections = []
        for frame_counter in range(video_len):
            if frame_counter in pred_dict["Frame"]:
                i = pred_dict["Frame"].index(frame_counter)
                ball_detections.append(
                    Ball(
                        frame=frame_counter,
                        xy=(pred_dict["X"][i], pred_dict["Y"][i]),
                        visibility=pred_dict["Visibility"][i]
                    )
                )
            else:
                print(f"{self.__str__()}: missing detection frame {frame_counter}")
                ball_detections.append(
                    Ball(
                        frame=frame_counter,
                        xy=(0.0, 0.0),
                        visibility=0,
                    )
                )

        # for i, frame_index in enumerate(pred_dict["Frame"]):
        #     if frame_counter == frame_index:
        #         ball_detections.append(
        #             Ball(
        #                 frame=frame_index,
        #                 xy=(pred_dict["X"][i], pred_dict["Y"][i]),
        #                 visibility=pred_dict["Visibility"][i]
        #             )
        #         )
            
        return ball_detections

        

            