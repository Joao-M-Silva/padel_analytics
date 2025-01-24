import numpy as np
import pandas as pd

from trackers.ball_tracker.kalman3d_tracking import KalmanFilter3DTracking


class Test3DFilter:
    def test_estimate_initial_state(self, court_model, ball_detections):
        k_filter = KalmanFilter3DTracking(court_model=court_model)
        dt = 1. / 30

        detections = ball_detections[:20]
        observations = [[ball['xy'][0], ball['xy'][1], t * dt] for t, ball in enumerate(detections)]

        x0 = k_filter.estimate_initial_state(observations)

        assert len(x0) == 7

    def test_filter(self, court_model, ball_detections):

        x0 = [np.float64(6.476107924720994), np.float64(7.8566629539172705), np.float64(2.738050487917544), np.float64(5.043514269843491), np.float64(15.64070648671098), np.float64(4.859266770187743), 1]

        filter = KalmanFilter3DTracking(court_model=court_model, x0=x0)

        filter.track(ball_detections)
        x, y, z, vx, vy, vz, _ = zip(*filter.states)

        print(pd.DataFrame(dict(x=x, y=y, z=z)))
        print(pd.DataFrame(dict(vx=vx, vy=vy, vz=vz)))
        assert len(x) == len(ball_detections)

        fig = filter.plot()
        # fig.show()
        # Save to file
        with open("../render.html", "w") as f:
            f.write(fig.to_html())

    def test_noise_sensitivity(self, court_model, ballistic_detections):
        """
        Make sure the noise sensitivity is adjusted so that the ball location does not
        jump too much even when the noise is high.
        """

        # Feed the model with a reasonable first guess, make sure it can track it
        x0 = np.array([
            court_model.width * 2 / 3, court_model.length * 1/3, 1.5,
            .5, 4, 1,
            1
        ])

        k_filter = KalmanFilter3DTracking(court_model=court_model, x0=x0)

        k_filter.track(ballistic_detections)
        x, y, z, vx, vy, vz, _ = zip(*k_filter.states)

        inferred_pos = pd.DataFrame(dict(x=x, y=y, z=z))
        inferred_vel = pd.DataFrame(dict(vx=vx, vy=vy, vz=vz))

        inferred_abs_vel = (inferred_vel ** 2).sum(axis=1) ** 0.5

        # Make sure velocities stay within the average range
        mean_vel = inferred_abs_vel.mean()
        std_vel = inferred_abs_vel.std()

        # k_filter.dump("test.html")

        # If the inferred position were too dependent on ball detection glitches, the velocity have high spikes
        assert all(inferred_abs_vel < mean_vel + 3 * std_vel)
