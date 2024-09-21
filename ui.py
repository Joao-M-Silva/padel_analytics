import cv2
import json
from utils import read_video

KEYPOINTS = []

# function to display the coordinates of 
# of the points clicked on the image  
def click_event(event, x, y, flags, params): 
  
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        KEYPOINTS.append((x, y))
  
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 
        cv2.imshow('image', img) 

if __name__ == "__main__":

    input_video_path = "./videos/trimmed_padel.mp4"
    # input_video_path = "./videos/FINAL A1PADEL MARBELLA MASTER  Tolito Aguirre  Alfonso vs Allemandi  Pereyra HIGHLIGHTS.mp4"
    frames, fps, w, h = read_video(input_video_path, max_frames=10)

    img = frames[0]
    cv2.imshow('image', img)

    # setting mouse handler for the image 
    # and calling the click_event() function 
    cv2.setMouseCallback('image', click_event) 

    # wait for a key to be pressed to exit 
    cv2.waitKey(0) 

    with open("source_keypoints.json", "w") as f:
        json.dump(KEYPOINTS, f)
  
    # close the window 
    cv2.destroyAllWindows() 