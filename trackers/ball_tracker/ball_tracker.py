from typing import Literal
from collections import deque
import json
from dataclasses import dataclass
from pathlib import Path
import math
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import torch

from trackers.ball_tracker.models import TrackNet, InpaintNet
from trackers.ball_tracker.dataset import BallTrajectoryDataset
from trackers.ball_tracker.predict import predict


def get_model(
    model_name: Literal["TrackNet", "InpaintNet"], 
    seq_len: int = None, 
    bg_mode: Literal["", "subtract", "subtract_concat", "concat"] = None,
):
    """ Create model by name and the configuration parameter.

        Args:
            model_name (str): type of model to create
                Choices:
                    - 'TrackNet': Return TrackNet model
                    - 'InpaintNet': Return InpaintNet model
            seq_len (int, optional): Length of TrackNet input sequence 
            bg_mode (str, optional): Background mode of TrackNet
                Choices:
                    - '': Return TrackNet with L x 3 input channels (RGB)
                    - 'subtract': Return TrackNet with L x 1 input channel 
                      (Difference frame)
                    - 'subtract_concat': Return TrackNet with L x 4 input channels
                      (RGB + Difference frame)
                    - 'concat': Return TrackNet with (L+1) x 3 input channels (RGB)

        Returns:
            model (torch.nn.Module): Model with specified configuration
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


def get_ensemble_weight(seq_len, eval_mode):
    """ Get weight for temporal ensemble.

        Args:
            seq_len (int): Length of input sequence
            eval_mode (str): Mode of temporal ensemble
                Choices:
                    - 'average': Return uniform weight
                    - 'weight': Return positional weight
        
        Returns:
            weight (torch.Tensor): Weight for temporal ensemble
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

def generate_inpaint_mask(pred_dict, th_h=30):
    """ Generate inpaint mask form predicted trajectory.

        Args:
            pred_dict (Dict): Prediction result
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
            th_h (float): Height threshold (pixels) for y coordinate
        
        Returns:
            inpaint_mask (List): Inpaint mask
    """
    y = np.array(pred_dict['Y'])
    vis_pred = np.array(pred_dict['Visibility'])
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


@dataclass
class Ball:

    """
    Definition of a ball 

    Attributes:
        frame: frame associated with the given ball detection
        xy: ball position coordinates
        visibility: 1 if the ball is visible in the given frame
        projection: ball position mini court projection 
    """

    frame: int
    xy: tuple[float, float]
    visibility: Literal[0, 1]
    projection: tuple[int, int] = None

    @classmethod
    def from_dict(cls, x: dict):
        return cls(**x)

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "xy": self.xy,
            "visibility": self.visibility,
        }
    
    def asint(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.xy)
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        cv2.circle(
            frame,
            tuple(int(x) for x in self.xy),
            6,
            (255, 255, 0),
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


class BallTracker:

    """
    MAX_SAMPLE_NUM is the maximum number of frames to sample for 
    generating median image.
    Note: its important to filter frames of interest before 
    feeding the video to the model
    """

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    EVAL_MODE: str = "weight"
    MAX_SAMPLE_NUM: int = 1800
    TRAJECTORY_LENGTH: int = 8
    
    HEIGHT: int = 288
    WIDTH: int = 512
    SIGMA: float = 2.5
    IMG_FORMAT = 'png'
    
    def __init__(
        self, 
        tracking_model_path: str,
        inpainting_model_path: str,
    ):
        self.DELTA_T: float = 1 / math.sqrt(self.HEIGHT**2 + self.WIDTH**2)
        self.COOR_TH = self.DELTA_T * 50

        tracknet_ckpt = torch.load(tracking_model_path)
        self.tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
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

    def load_detections(self, path: str | Path) -> list[Ball]:

        print("Loading Ball Detections")

        with open(path, "r") as f:
            parsable_ball_detections = json.load(f)

        print("Done.")

        return [
            Ball.from_dict(ball_detection)
            for ball_detection in parsable_ball_detections
        ]

    def detect_frames(
        self, 
        frames: list[np.ndarray],
        width: int,
        height: int,
        batch_size: int = 1,
        save_path: str | Path = None,
        load_path: str | Path = None,
    ) -> list[Ball]:
        
        if load_path is not None:
            ball_detections = self.load_detections(load_path)

            return ball_detections
        
        self.tracknet.to(self.DEVICE)
        if self.inpaintnet is not None:
            print(self.DEVICE)
            self.inpaintnet.to(self.DEVICE)

        w_scaler, h_scaler = (
            width / self.WIDTH, 
            height / self.HEIGHT,
        )

        img_scaler = (w_scaler, h_scaler)

        tracknet_pred_dict = {
            'Frame':[], 
            'X':[], 
            'Y':[], 
            'Visibility':[], 
            'Inpaint_Mask':[],
            'Img_scaler': (w_scaler, h_scaler), 
            'Img_shape': (width, height),
        }

        seq_len = self.tracknet_seq_len
        print("here")
        dataset = BallTrajectoryDataset(
            seq_len=seq_len, 
            sliding_step=1, 
            data_mode='heatmap', 
            bg_mode=self.bg_mode,
            frame_arr=np.array(frames),
            HEIGHT=self.HEIGHT,
            WIDTH=self.WIDTH,
            SIGMA=self.SIGMA,
            IMG_FORMAT=self.IMG_FORMAT,
        )
        print("HERE")
        
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False, 
            drop_last=False,
        )  # num_workers=num_workers
        video_len = len(frames)

        # Init prediction buffer params
        num_sample, sample_count = video_len - seq_len + 1, 0
        buffer_size = seq_len - 1
        batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
        frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
        y_pred_buffer = torch.zeros((buffer_size, seq_len, self.HEIGHT, self.WIDTH), dtype=torch.float32)
        weight = get_ensemble_weight(seq_len, self.EVAL_MODE)
        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().to(self.DEVICE)
            b_size, seq_len = i.shape[0], i.shape[1]
            with torch.no_grad():
                y_pred = self.tracknet(x).detach().cpu()
            
            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
            ensemble_y_pred = torch.empty((0, 1, self.HEIGHT, self.WIDTH), dtype=torch.float32)

            for b in range(b_size):
                if sample_count < buffer_size:
                    # Imcomplete buffer
                    y_pred = y_pred_buffer[batch_i+b, frame_i].sum(0) / (sample_count+1)
                else:
                    # General case
                    y_pred = (y_pred_buffer[batch_i+b, frame_i] * weight[:, None, None]).sum(0)
                
                ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, self.HEIGHT, self.WIDTH)), dim=0)
                sample_count += 1

                if sample_count == num_sample:
                    # Last batch
                    y_zero_pad = torch.zeros((buffer_size, seq_len, self.HEIGHT, self.WIDTH), dtype=torch.float32)
                    y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)

                    for f in range(1, seq_len):
                        # Last input sequence
                        y_pred = y_pred_buffer[batch_i+b+f, frame_i].sum(0) / (seq_len-f)
                        ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, self.HEIGHT, self.WIDTH)), dim=0)

            # Predict
            tmp_pred = predict(
                ensemble_i, 
                y_pred=ensemble_y_pred, 
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
            tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=height*0.05)
            inpaint_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

            # Create dataset with overlap sampling for temporal ensemble
            dataset = BallTrajectoryDataset(
                seq_len=seq_len, 
                sliding_step=1, 
                data_mode='coordinate', 
                pred_dict=tracknet_pred_dict,
                HEIGHT=self.HEIGHT,
                WIDTH=self.WIDTH,
                SIGMA=self.SIGMA,
                IMG_FORMAT=self.IMG_FORMAT,
            )
            data_loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                drop_last=False,
            ) # num_workers=num_workers, 

            weight = get_ensemble_weight(seq_len, self.EVAL_MODE)

            # Init buffer params
            num_sample, sample_count = len(dataset), 0
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
            frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
            coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
            
            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                b_size = i.shape[0]
                with torch.no_grad():
                    coor_inpaint = self.inpaintnet(coor_pred.cuda(), inpaint_mask.cuda()).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask)
                
                # Thresholding
                th_mask = ((coor_inpaint[:, :, 0] < self.COOR_TH) & (coor_inpaint[:, :, 1] < self.COOR_TH))
                coor_inpaint[th_mask] = 0.

                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)
                
                for b in range(b_size):
                    if sample_count < buffer_size:
                        # Imcomplete buffer
                        coor_inpaint = coor_inpaint_buffer[batch_i+b, frame_i].sum(0)
                        coor_inpaint /= (sample_count+1)
                    else:
                        # General case
                        coor_inpaint = (coor_inpaint_buffer[batch_i+b, frame_i] * weight[:, None]).sum(0)
                    
                    ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                    ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)
                    sample_count += 1

                    if sample_count == num_sample:
                        # Last input sequence
                        coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                        coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)
                        
                        for f in range(1, seq_len):
                            coor_inpaint = coor_inpaint_buffer[batch_i+b+f, frame_i].sum(0)
                            coor_inpaint /= (seq_len-f)
                            ensemble_i = torch.cat((ensemble_i, i[-1][f].view(1, 1, 2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)

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
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
                # Update buffer, keep last predictions for ensemble in next iteration
                coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]

        pred_dict = inpaint_pred_dict if self.inpaintnet is not None else tracknet_pred_dict
        
        ball_detections = []
        for i, frame_index in enumerate(pred_dict["Frame"]):
            ball_detections.append(
                Ball(
                    frame=frame_index,
                    xy=(pred_dict["X"][i], pred_dict["Y"][i]),
                    visibility=pred_dict["Visibility"][i]
                )
            )

        if save_path is not None:

            print("Saving Ball Detections...")

            parsable_ball_detections = [
                ball_detection.to_dict()
                for ball_detection in ball_detections
            ]

            with open(save_path, "w") as f:
                json.dump(
                    parsable_ball_detections,
                    f,
                )
            
            print("Done.")
        
        self.tracknet.to("cpu")
        if self.inpaintnet is not None:
            self.inpaintnet.to("cpu")
        
        return ball_detections
    
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
        
   