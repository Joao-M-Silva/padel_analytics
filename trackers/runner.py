""" 
Implementation of a runner to extract results from an arbitrary list of trackers 
"""

from typing import Iterable, Optional
from tqdm import tqdm
import timeit
from pathlib import Path
import numpy as np
import cv2
import supervision as sv

from trackers.tracker import Object, TrackingResults, Tracker


class TrackingRunner:

    """
    Abstraction that implements a memory efficient pipeline to run
    a sequence of trackers over a sequence of video frames

    Attributes:
        trackers: sequence of trackers of interest
        video_path: source video path
        inference_path: path where to save the inference results
        start: indicates the starting position from which video should generate frames
        stride: indicates the interval at which frames are returned
        end: indicates the ending position at which video should stop generating frames.
             If None, video will be read to the end.     
    """

    def __init__(
        self, 
        trackers: list[Tracker],
        video_path: str | Path,
        inference_path: str | Path,
        start: int = 0,
        end: Optional[int] = None,
    ) -> None:
    
        self.video_path = video_path
        self.inference_path = inference_path
        self.start = start
        self.stride = 1
        self.end = end
        self.video_info = sv.VideoInfo.from_video_path(video_path=video_path)

        if self.end is None:
            self.total_frames = self.video_info.total_frames
        else:
            self.total_frames = self.end - self.start

        self.trackers = {
            str(tracker): tracker.video_info_post_init(self.video_info)
            for tracker in trackers
        }
    
    def restart(self) -> None:
        """
        Restart all trackers
        """
        for tracker in self.trackers.values():
            tracker.restart()

    def draw(self) -> None:
        """
        Draw tracker results accross all video frames
        """

        print(f"Writing results into {str(self.inference_path)}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.inference_path,
            fourcc,
            float(self.video_info.fps),
            self.video_info.resolution_wh,
        )

        frame_generator = sv.get_video_frames_generator(
            self.video_path,
            start=self.start,
            stride=self.stride,
            end=self.end,
        )

        for frame_index, frame in tqdm(enumerate(frame_generator)):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for tracker in self.trackers.values():
                prediction = tracker.results[frame_index]
                frame_rgb = prediction.draw(frame_rgb, **tracker.draw_kwargs())

            out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        
        out.release()

        print("Done.")

    def run(self) -> None:
        """
        Run trackers object prediction for every frame in the frame generator

        Parameters:
            drop_last: True to drop the last sample if its incomplete
        """

        print(f"Running {self.total_frames} frames")

        for tracker in self.trackers.values():

            if len(tracker) != 0:
                print(f"{tracker.__str__()}: {len(tracker)} predictions stored")
                if len(tracker) == self.total_frames:
                    print(
                        f"""{tracker.__str__()}: \
                        match between number of predictions and total frames 
                        """
                    )
                    continue
                else:
                    print(
                        f"""{tracker.__str__()}: \
                        unmatch between number of predictions and total frames 
                        """
                    )
                    tracker.restart()
                    print(f"{tracker.__str__()}: WARNING restarted tracker")

            tracker.to(tracker.DEVICE)
            print(f"{str(tracker)}: Running on {tracker.DEVICE} ...")

            frame_generator = sv.get_video_frames_generator(
                self.video_path,
                start=self.start,
                stride=self.stride,
                end=self.end,
            )

            t0 = timeit.default_timer()
            tracker.predict_and_update(
                frame_generator, 
                total_frames=self.total_frames,
            )
            t1 = timeit.default_timer()

            tracker.to("cpu")

            print(f"{str(tracker)}: {t1 - t0} inference time.")

            tracker.save_predictions()
        
        self.draw()

        

    

    


        



    
    


