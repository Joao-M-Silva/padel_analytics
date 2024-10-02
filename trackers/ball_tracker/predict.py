from typing import Union
import numpy as np
import cv2
import torch


def predict_location(heatmap: np.array):
    """ Get coordinates from the heatmap.

        Args:
            heatmap (numpy.ndarray): A single heatmap with shape (H, W)

        Returns:
            x, y, w, h (Tuple[int, int, int, int]): bounding box of the the bounding box with max area
    """
    if np.amax(heatmap) == 0:
        # No respond in heatmap
        return 0, 0, 0, 0
    else:
        # Find all respond area in the heapmap
        (cnts, _) = cv2.findContours(
            heatmap.copy(), 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        rects = [cv2.boundingRect(ctr) for ctr in cnts]

        # Find largest area amoung all contours
        max_area_idx = 0
        max_area = rects[0][2] * rects[0][3]
        for i in range(1, len(rects)):
            area = rects[i][2] * rects[i][3]
            if area > max_area:
                max_area_idx = i
                max_area = area

        x, y, w, h = rects[max_area_idx]

        return x, y, w, h
    

def to_img(image):
    """ Convert the normalized image back to image format.

        Args:
            image (numpy.ndarray): Images with range in [0, 1]

        Returns:
            image (numpy.ndarray): Images with range in [0, 255]
    """

    image = image * 255
    image = image.astype('uint8')
    return image


def to_img_format(input, WIDTH: int, HEIGHT: int, num_ch=1):
    """ Helper function for transforming model input sequence format to image sequence format.

        Args:
            input (numpy.ndarray): model input with shape (N, L*C, H, W)
            num_ch (int): Number of channels of each frame.

        Returns:
            (numpy.ndarray): Image sequences with shape (N, L, H, W) or (N, L, H, W, 3)
    """

    assert len(input.shape) == 4, 'Input must be 4D tensor.'
    
    if num_ch == 1:
        # (N, L, H ,W)
        return input
    else:
        # (N, L*C, H ,W)
        input = np.transpose(input, (0, 2, 3, 1)) # (N, H ,W, L*C)
        seq_len = int(input.shape[-1]/num_ch)
        img_seq = np.array([]).reshape(0, seq_len, HEIGHT, WIDTH, 3) # (N, L, H, W, 3)
        # For each sample in the batch
        for n in range(input.shape[0]):
            frame = np.array([]).reshape(0, HEIGHT, WIDTH, 3)
            # Get each frame in the sequence
            for f in range(0, input.shape[-1], num_ch):
                img = input[n, :, :, f:f+3]
                frame = np.concatenate((frame, img.reshape(1, HEIGHT, WIDTH, 3)), axis=0)
            img_seq = np.concatenate((img_seq, frame.reshape(1, seq_len, HEIGHT, WIDTH, 3)), axis=0)
        
        return img_seq



def predict(indices, WIDTH: int, HEIGHT: int, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """ Predict coordinates from heatmap or inpainted coordinates. 

        Args:
            indices (torch.Tensor): indices of input sequence with shape (N, L, 2)
            y_pred (torch.Tensor, optional): predicted heatmap sequence with shape (N, L, H, W)
            c_pred (torch.Tensor, optional): predicted inpainted coordinates sequence with shape (N, L, 2)
            img_scaler (Tuple): image scaler (w_scaler, h_scaler)

        Returns:
            pred_dict (Dict): dictionary of predicted coordinates
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
    """

    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy()if torch.is_tensor(indices) else indices.numpy()
    
    # Transform input for heatmap prediction
    if y_pred is not None:
        y_pred = y_pred > 0.5
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred, WIDTH=WIDTH, HEIGHT=HEIGHT) # (N, L, H, W)
    
    # Transform input for coordinate prediction
    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if c_pred is not None:
                    # Predict from coordinate
                    c_p = c_pred[n][f]
                    cx_pred, cy_pred = int(c_p[0] * WIDTH * img_scaler[0]), int(c_p[1] * HEIGHT* img_scaler[1]) 
                elif y_pred is not None:
                    # Predict from heatmap
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location(to_img(y_p))
                    cx_pred, cy_pred = int(bbox_pred[0]+bbox_pred[2]/2), int(bbox_pred[1]+bbox_pred[3]/2)
                    cx_pred, cy_pred = int(cx_pred*img_scaler[0]), int(cy_pred*img_scaler[1])
                else:
                    raise ValueError('Invalid input')
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                prev_f_i = f_i
            else:
                break
    
    return pred_dict    


def predict_modified(
    WIDTH: int, 
    HEIGHT: int, 
    y_pred: Union[torch.Tensor, np.array] = None, 
    c_pred: Union[torch.Tensor, np.array] = None, 
    img_scaler: tuple[float, float] = (1.0, 1.0),
    threshold: float = 0.5,
) -> dict:
    """ Predict coordinates from heatmap or inpainted coordinates. 

        Args:
            y_pred (torch.Tensor, optional): predicted heatmap sequence with shape (N, L, H, W)
            c_pred (torch.Tensor, optional): predicted inpainted coordinates sequence with shape (N, L, 2)
            img_scaler (Tuple): image scaler (w_scaler, h_scaler)

        Returns:
            pred_dict (Dict): dictionary of predicted coordinates
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
    """

    pred_dict = {
        'x': [], 
        'y': [], 
        'visibility': []
    }
    
    # Transform input for heatmap prediction
    if y_pred is not None:
        y_pred = y_pred > threshold
        y_pred = (
            y_pred.detach().cpu().numpy() 
            if torch.is_tensor(y_pred) 
            else y_pred
        )
    
    # Transform input for coordinate prediction
    if c_pred is not None:
        c_pred = (
            c_pred.detach().cpu().numpy() 
            if torch.is_tensor(c_pred) 
            else c_pred
        )

    number_preds = y_pred.shape[0]
    for n in range(number_preds):
        if c_pred is not None:
            # Predict from coordinate
            c_p = c_pred[n][0]
            cx_pred, cy_pred = (
                int(c_p[0] * WIDTH * img_scaler[0]), 
                int(c_p[1] * HEIGHT * img_scaler[1]),
            )
        elif y_pred is not None:
            # Predict from heatmap
            y_p = y_pred[n][0]
            bbox_pred = predict_location(to_img(y_p))
            cx_pred, cy_pred = (
                int(bbox_pred[0]+bbox_pred[2]/2), 
                int(bbox_pred[1]+bbox_pred[3]/2),
            )
            cx_pred, cy_pred = (
                int(cx_pred*img_scaler[0]), 
                int(cy_pred*img_scaler[1]),
            )
        else:
            raise ValueError('Invalid input')
        
        viz_pred = 0 if (cx_pred == 0 and cy_pred == 0) else 1
        pred_dict["x"].append(cx_pred)
        pred_dict["y"].append(cy_pred)
        pred_dict["visibility"].append(viz_pred)
    
    return pred_dict
