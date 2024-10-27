import pandas as pd

from trackers.ball_tracker.kalman3d_tracking import KalmanFilter3DTracking


class Test3DFilter:

    def test_filter(self, court_model, ball_detections):

        filter = KalmanFilter3DTracking(court_model=court_model)

        filter.track(ball_detections)
        x, y, z, vx, vy, vz, _ = zip(*filter.states)

        print(pd.DataFrame(dict(x=x, y=y, z=z)))
        print(pd.DataFrame(dict(vx=vx, vy=vy, vz=vz)))
        assert len(x) == len(ball_detections)

        fig = filter.plot()
        fig.show()
        # Save to file
        with open("../render.html", "w") as f:
            f.write(fig.to_html())
