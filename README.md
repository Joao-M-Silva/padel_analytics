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
#### 2. Setup virtual environment.
```
conda create -n python=3.12 <yourenv> pip
conda activate <yourenv>
pip install -r requirements.txt
```
#### 3. Install pytorch <https://pytorch.org/get-started/locally/>.
#### 4. Download weights.
   In order to have access to the models' weights, please email me at <jsilvawasd@hotmail.com> describing your motivation and goals regarding this project. I am currenly trying to evolve to a point where this project is monetizable, as so,    the access to the models' weights is limited to possible collaborations. I encourage you to train your own models and get this framework to the next level!

# Inference
At the root of this repo edit the file config.py and run:
````
python main.py
````

![inference](https://github.com/user-attachments/assets/5a7432ff-35a6-4db4-acc2-cdb760b4bd8d)




