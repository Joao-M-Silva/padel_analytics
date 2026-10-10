# Padel Analytics
![padel analytics](https://github.com/user-attachments/assets/f66e6141-6ad7-48ca-b363-f539af0782ca)

This repository applies computer vision techniques to extract valuable insights from a padel game recording like:
- Position and velocity of each player;
- Position and velocity of the ball;
- 2D game projection;
- Heatmaps;
- Ball velocity associated with distinct strokes;
- Player error rate.

To do so, several computer vision models where trained in order to:
1. Track the position of each individual players;
2. Players pose estimation with 13 degrees of freedom;
3. Players pose classification (e.g. backhand/forehand volley, bandeja, topspin smash, etc);
4. Predict ball hits.

The goal of this project is to provide precise and robust analytics using only a padel game recording. This implementation can be used to:
1. Upgrade live broadcasts providing interesting data to be shared with the audience or to be stored in a database for future analysis;
2. Generate precious insights to be used by padel coachs or players to enhance their path of continuous improvement.

# Setup
#### 1. Clone this repository.
#### 2. Set up a virtual environment.
Install Python 3.12, then run these commands from the repository root on macOS or Linux:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```
On Windows (PowerShell), use `py -3.12 -m venv .venv` and `.venv\Scripts\Activate.ps1` instead of the first two commands.

#### 3. Install dependencies.
With the environment activated, install `torch` and `torchvision` using the command from the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for your operating system and compute platform. Then install the remaining dependencies:
```bash
python -m pip install -r requirements.txt
```
The Streamlit app (`app.py`) also needs the [FFmpeg executable](https://ffmpeg.org/download.html) available on your `PATH`. Install it using your system package manager:

On Linux: `sudo apt update && sudo apt install ffmpeg`
On macOS: `brew install ffmpeg`

Confirm the installationwith:
```bash
ffmpeg -version
```

Plotly PNG export also needs Chrome or Chromium installed; see the [Plotly image export instructions](https://plotly.com/python/static-image-export/).

#### 4. Download weights.
   The current model weights used are available here https://drive.google.com/drive/folders/1joO7w1Am7B418SIqGBq90YipQl81FMzh?usp=drive_link. Configure the config.py file with your own model checkpoints paths. 
# Inference
At the root of this repo, edit the file config.py accordingly and run:
````
python main.py
````
#### VRAM requirements
Using the default batch sizes one will need to have at least 8GB of VRAM. Reduce batch sizes editing the config.py file according to your needs. 
#### Implementation details
Currently this implementation assumes a fixed camera setup. As a result, a UI for selecting court keypoints will pop up asking you to select 12 unique court keypoints that are further used for homographic computations. A video describing the keypoints selection is available at `./examples/videos/select_keypoints.mp4`. Please refer to main.py lines 24-38 where a diagram showcasing keypoints numeration is drawn.
#### Keypoints selection
![select_keypoints_animation](https://github.com/user-attachments/assets/3c15131f-9943-477b-adeb-782cc32e8946)
#### Inference results
![inference](https://github.com/user-attachments/assets/5a7432ff-35a6-4db4-acc2-cdb760b4bd8d)

# Collaborations
I am currently looking for collaborations to uplift this project to new heights. If you are interested feel free to e-mail me at jsilvawasd@hotmail.com.






