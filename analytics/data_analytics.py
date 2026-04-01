from typing import Optional
from dataclasses import dataclass
from copy import deepcopy
import pandas as pd
import numpy as np
import functools


class InvalidDataPoint(Exception):
    pass


@dataclass
class PlayerPosition:

    """
    Player position (meters) in a given frame
    """

    id: int
    position: tuple[float, float]

    def __post_init__(self):
        assert isinstance(self.position[0], float)
        assert isinstance(self.position[1], float)

    @property
    def key(self) -> str:
        return f"player{self.id}"

@dataclass
class DataPoint:

    """
    Tracker objects data collected in a given frame

    Attributes:
        frame: frame of interest
        players_position: players position (meters) in the given frame
        ball_position: ball position (meters) in the given frame, or None
    """

    frame: int = None
    players_position: list[PlayerPosition] = None
    ball_position: Optional[tuple[float, float]] = None

    def validate(self) -> None:

        if self.frame is None:
            raise InvalidDataPoint("Unknown frame")
        
        if self.players_position is None:
            print("data_analytics: WARNING(Missing players position)")
            return None
        
        players_ids = []
        valid_positions = []
        for player_pos in self.players_position:
            player_id = player_pos.id
            if player_id in (1, 2, 3, 4):
                players_ids.append(player_id)
                valid_positions.append(player_pos)
        self.players_position = valid_positions

        if len(players_ids) != len(set(players_ids)):
            raise InvalidDataPoint("N-plicate player id")
        
        if len(self.players_position) != 4:
            number_players_missing = 4 - len(self.players_position)
            print(f"{number_players_missing} player/s missing")
        
    def add_player_position(self, player_position: PlayerPosition):
        if self.players_position is None:
            self.players_position = [player_position]
        else:
            self.players_position.append(player_position)

    def add_ball_position(self, position: tuple[float, float]):
        self.ball_position = position

    def sort_players_position(self) -> Optional[list[PlayerPosition]]:
        if self.players_position:
            players_position = sorted(
                self.players_position, 
                key=lambda x: x.id,
            )
            return players_position
        
        print("data_analytics: impossible to sort, missing players position")
        return None

class DataAnalytics:

    """
    Tracker objects data collector 
    """

    def __init__(self):
        self.frames = [0]
        self.current_datapoint = DataPoint(frame=self.frames[-1])
        self.datapoints: list[DataPoint] = []

    def restart(self) -> None:
        self.__init__()

    @classmethod
    def from_dict(cls, data: dict):
        frames = data["frame"]
        instance = cls()
        instance.frames = frames

        datapoints = []
        for i in range(len(frames)):
            frame = frames[i]
            players_position = []
            for player_id in (1, 2, 3, 4):
                if (
                    data[f"player{player_id}_x"][i] is None
                    or 
                    data[f"player{player_id}_y"][i] is None
                ):
                    continue

                players_position.append(
                    PlayerPosition(
                        id=player_id,
                        position=(
                            data[f"player{player_id}_x"][i],
                            data[f"player{player_id}_y"][i],
                        )
                    )   
                )

            datapoints.append(
                DataPoint(
                    frame=frame, 
                    players_position=players_position if players_position else None,
                )
            )
        
        for i in range(len(frames)):
            frame = frames[i]
            ball_x = data.get("ball_x", [None] * len(frames))[i]
            ball_y = data.get("ball_y", [None] * len(frames))[i]
            if ball_x is not None and ball_y is not None:
                datapoints[i].ball_position = (float(ball_x), float(ball_y))

        instance.datapoints = datapoints
        instance.current_datapoint = None

        return instance
    
    def into_dict(self) -> dict[str, list]:
        data = {
            "frame": [],
            "player1_x": [],
            "player1_y": [],
            "player2_x": [],
            "player2_y": [],
            "player3_x": [],
            "player3_y": [],
            "player4_x": [],
            "player4_y": [],
            "ball_x": [],
            "ball_y": [],
        }

        for datapoint in self.datapoints:
            data["frame"].append(datapoint.frame)
            number_frames = len(data["frame"])

            players_position = datapoint.sort_players_position()
            if players_position:
                for player_position in players_position:
                    data[f"{player_position.key}_x"].append(
                        player_position.position[0]
                    )
                    data[f"{player_position.key}_y"].append(
                        player_position.position[1]
                    )

            if datapoint.ball_position is not None:
                data["ball_x"].append(datapoint.ball_position[0])
                data["ball_y"].append(datapoint.ball_position[1])

            # Append missing values
            for k, v in data.items():
                if len(v) < number_frames:
                    data[k].append(None)

        print("data_analytics: missing values")
        for k, v in data.items():
            print(f"data_analytics: {k} - {len([x for x in v if x is None])}/{len(v)}")

        return data

    def __len__(self) -> int:
        return len(self.frames)

    def update(self):
        self.current_datapoint.validate()
        self.datapoints.append(self.current_datapoint)
        self.current_datapoint = DataPoint(frame=self.frames[-1])
    
    def step(self, x: int = 1) -> None:
        new_frame = self.frames[-1] + 1

        assert new_frame not in self.frames

        self.frames.append(new_frame)
        self.update()

    def add_player_position(
        self,
        id: int,
        position: tuple[float, float],
    ):
        self.current_datapoint.add_player_position(
            PlayerPosition(
                id=id,
                position=position,
            )
        )

    def add_ball_position(self, position: tuple[float, float]):
        self.current_datapoint.add_ball_position(position)

    def into_dataframe(self, fps: int) -> pd.DataFrame:
        """
        Retrieves a dataframe with additional features
        """

        def norm(x: float, y: float) -> float:
            return np.sqrt(x**2 + y**2)

        def calculate_distance(row, player_id: int):
            return norm(
                row[f"player{player_id}_deltax1"], 
                row[f"player{player_id}_deltay1"], 
            )
        
        def calculate_norm_velocity(row, player_id: int, frame_interval: int) -> float:
            return norm(
                row[f"player{player_id}_Vx{frame_interval}"],
                row[f"player{player_id}_Vy{frame_interval}"],
            )

        def calculate_norm_acceleration(row, player_id: int, frame_interval: int) -> float:
            return norm(
                row[f"player{player_id}_Ax{frame_interval}"],
                row[f"player{player_id}_Ay{frame_interval}"],
            )

        frame_intervals = (1, 2, 3, 4)
        player_ids = (1, 2, 3, 4)

        df = pd.DataFrame(self.into_dict())
        df["time"] = df["frame"] * (1/fps)

        # Coerce player and ball position columns to float so None becomes NaN and
        # arithmetic operations (diff, eval) work on frames with missing data
        for player_id in player_ids:
            for pos in ("x", "y"):
                df[f"player{player_id}_{pos}"] = pd.to_numeric(
                    df[f"player{player_id}_{pos}"], errors="coerce"
                )
        for pos in ("x", "y"):
            df[f"ball_{pos}"] = pd.to_numeric(df[f"ball_{pos}"], errors="coerce")

        # Ball velocity (frame-interval=1 only)
        df["delta_time1"] = df["time"].diff(1)
        for pos in ("x", "y"):
            df[f"ball_delta{pos}1"] = df[f"ball_{pos}"].diff(1)
        df["ball_Vx1"] = df["ball_deltax1"] / df["delta_time1"]
        df["ball_Vy1"] = df["ball_deltay1"] / df["delta_time1"]
        df["ball_Vnorm1"] = df.apply(
            lambda row: norm(row["ball_Vx1"], row["ball_Vy1"]),
            axis=1,
        )

        for frame_interval in frame_intervals:
            # Time in seconds between each frame for a given frame interval
            df[f"delta_time{frame_interval}"] = df["time"].diff(frame_interval)
            for player_id in player_ids:
                for pos in ("x", "y"):
                    # Displacement in x and y for each of the players 
                    # for a given time interval
                    df[
                        f"player{player_id}_delta{pos}{frame_interval}"
                    ] = df[f"player{player_id}_{pos}"].diff(frame_interval)

                    # Velocity in x and y for each of the players 
                    # for a given time interval
                    eval_string_velocity = f"""
                    player{player_id}_delta{pos}{frame_interval} / delta_time{frame_interval}
                    """
                    df[f"player{player_id}_V{pos}{frame_interval}"] = df.eval(
                        eval_string_velocity,
                    )

                    # Velocity difference in x and y for each of the players 
                    # for a given time interval
                    df[
                        f"player{player_id}_deltaV{pos}{frame_interval}"
                    ] = df[f"player{player_id}_V{pos}{frame_interval}"].diff(frame_interval)

                    # Acceleration in x and y for each of the players
                    # for a given time interval
                    eval_string_acceleration = f"""
                    player{player_id}_deltaV{pos}{frame_interval} / delta_time{frame_interval}
                    """
                    df[f"player{player_id}_A{pos}{frame_interval}"] = df.eval(
                        eval_string_acceleration,
                    )
                
                # Calculate player distance in between frames
                df[f"player{player_id}_distance"] = df.apply(
                    functools.partial(calculate_distance, player_id=player_id),
                    axis=1,
                )

                # Calculate norm velocity for each of the players
                # for a given time interval
                df[f"player{player_id}_Vnorm{frame_interval}"] = df.apply(
                    functools.partial(
                        calculate_norm_velocity, 
                        player_id=player_id,
                        frame_interval=frame_interval,
                    ),
                    axis=1,
                )

                # Calculate norm acceleration for each of the players
                # for a given time interval
                df[f"player{player_id}_Anorm{frame_interval}"] = df.apply(
                    functools.partial(
                        calculate_norm_acceleration, 
                        player_id=player_id,
                        frame_interval=frame_interval,
                    ),
                    axis=1,
                )
        
        return df


def zone_breakdown(df: pd.DataFrame, player_id: int) -> dict:
    """
    Returns % time spent in each court zone for a given player.

    Zones (based on absolute y distance from net):
        front:      |y| < 3 m  (net zone)
        transition: 3 <= |y| < 6 m  (mid-court)
        back:       |y| >= 6 m  (back court)
    """
    y = df[f"player{player_id}_y"].dropna()
    if len(y) == 0:
        return {"front": 0.0, "transition": 0.0, "back": 0.0}

    abs_y = y.abs()
    total = len(y)
    front = int((abs_y < 3).sum())
    transition = int(((abs_y >= 3) & (abs_y < 6)).sum())
    back = int((abs_y >= 6).sum())

    return {
        "front": round(100 * front / total, 1),
        "transition": round(100 * transition / total, 1),
        "back": round(100 * back / total, 1),
    }


def partner_synchrony(
    df: pd.DataFrame,
    player_a: int,
    player_b: int,
    window: int = 60,
) -> dict:
    """
    Computes synchrony metrics for a player pair over a rolling window.

    Returns:
        vertical_sync:       mean rolling Pearson r of Vy (forward/backward movement)
        horizontal_sync:     mean rolling Pearson r of Vx (side-to-side movement)
        avg_formation_width: median |x_a − x_b| in metres
        rolling_vertical_sync:   list of per-frame rolling r values (for charts)
        rolling_horizontal_sync: list of per-frame rolling r values (for charts)
    """
    vya = df[f"player{player_a}_Vy1"]
    vyb = df[f"player{player_b}_Vy1"]
    vxa = df[f"player{player_a}_Vx1"]
    vxb = df[f"player{player_b}_Vx1"]
    xa = df[f"player{player_a}_x"]
    xb = df[f"player{player_b}_x"]

    rolling_v = vya.rolling(window, min_periods=2).corr(vyb)
    rolling_h = vxa.rolling(window, min_periods=2).corr(vxb)

    vertical_sync = rolling_v.mean()
    horizontal_sync = rolling_h.mean()
    formation_width = (xa - xb).abs().median()

    def _safe(v):
        return round(float(v), 3) if not np.isnan(v) else None

    return {
        "vertical_sync": _safe(vertical_sync),
        "horizontal_sync": _safe(horizontal_sync),
        "avg_formation_width": _safe(formation_width) if formation_width is not None and not np.isnan(formation_width) else None,
        "rolling_vertical_sync": rolling_v.tolist(),
        "rolling_horizontal_sync": rolling_h.tolist(),
    }


# ── Coaching KPI functions ────────────────────────────────────────────────────


def net_approach_count(df: pd.DataFrame, player_id: int) -> int:
    """
    Count how many times a player entered the net zone (|y| < 2 m).
    A new approach is counted each time the player crosses into the net zone
    after being outside it.
    """
    y = pd.to_numeric(df[f"player{player_id}_y"], errors="coerce")
    in_net = (y.abs() < 2)
    # Count transitions from outside → inside net zone
    return int((in_net & ~in_net.shift(1, fill_value=False)).sum())


def time_in_nomansland_pct(df: pd.DataFrame, player_id: int) -> float:
    """
    Percentage of frames where player is in the transition zone (3 <= |y| < 6 m).
    High values indicate poor positioning discipline.
    """
    y = pd.to_numeric(df[f"player{player_id}_y"], errors="coerce").dropna()
    if len(y) == 0:
        return 0.0
    abs_y = y.abs()
    return round(100 * float(((abs_y >= 3) & (abs_y < 6)).sum()) / len(y), 1)


def change_of_direction_count(
    df: pd.DataFrame, player_id: int, threshold: float = 0.3
) -> int:
    """
    Count lateral direction reversals above a speed threshold (m/s).
    High count = explosive, active lateral movement.
    Low count = mostly linear or passive movement.
    """
    vx = pd.to_numeric(df[f"player{player_id}_Vx1"], errors="coerce")
    moving = vx.abs() > threshold
    sign = np.sign(vx)
    reversals = (sign != sign.shift(1)) & moving & moving.shift(1, fill_value=False)
    return int(reversals.sum())


def lateral_bias(df: pd.DataFrame, player_id: int) -> float:
    """
    Mean lateral velocity (Vx) as a bias indicator.
    Positive = predominantly moving right; negative = predominantly moving left.
    Near zero = balanced lateral coverage.
    """
    vx = pd.to_numeric(df[f"player{player_id}_Vx1"], errors="coerce")
    result = vx.mean()
    return round(float(result), 3) if not np.isnan(result) else 0.0


def recovery_speed(df: pd.DataFrame, player_id: int) -> float:
    """
    Mean speed (m/s) in frames immediately after the player exits the net zone
    (first 30 frames after leaving |y| < 2). Proxy for how quickly they recover
    to a defensive position after attacking the net.
    """
    y = pd.to_numeric(df[f"player{player_id}_y"], errors="coerce")
    vnorm = pd.to_numeric(df[f"player{player_id}_Vnorm1"], errors="coerce")
    in_net = y.abs() < 2
    exit_net = (~in_net) & in_net.shift(1, fill_value=False)

    speeds = []
    exit_indices = df.index[exit_net].tolist()
    for idx in exit_indices:
        loc = df.index.get_loc(idx)
        window_end = min(loc + 30, len(df))
        speeds.extend(vnorm.iloc[loc:window_end].dropna().tolist())

    if not speeds:
        return 0.0
    return round(float(np.mean(speeds)), 3)


def peak_sprint_count(
    df: pd.DataFrame, player_id: int, threshold: float = 5.5
) -> int:
    """
    Count the number of frames where player speed exceeds threshold (m/s).
    Proxy for explosive sprint frequency / physical conditioning.
    Default threshold 5.5 m/s ≈ 20 km/h (sprinting category).
    """
    vnorm = pd.to_numeric(df[f"player{player_id}_Vnorm1"], errors="coerce")
    return int((vnorm > threshold).sum())


def coaching_kpis(df: pd.DataFrame, player_id: int) -> dict:
    """
    Compute all coaching KPIs for a single player. Returns a flat dict.
    """
    return {
        "net_approach_count": net_approach_count(df, player_id),
        "time_in_nomansland_pct": time_in_nomansland_pct(df, player_id),
        "change_of_direction_count": change_of_direction_count(df, player_id),
        "lateral_bias": lateral_bias(df, player_id),
        "recovery_speed": recovery_speed(df, player_id),
        "peak_sprint_count": peak_sprint_count(df, player_id),
    }
