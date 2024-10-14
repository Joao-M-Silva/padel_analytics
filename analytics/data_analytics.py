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
    """

    frame: int = None
    players_position: list[PlayerPosition] = None

    def validate(self) -> None:

        if self.frame is None:
            raise InvalidDataPoint("Unknown frame")
        
        if self.players_position is None:
            print("data_analytics: WARNING(Missing players position)")
            return None
        
        players_ids = []
        for i, player_pos in enumerate(deepcopy(self.players_position)):
            player_id = player_pos.id

            if player_id in (1, 2, 3, 4):
                players_ids.append(player_id)
            else:
                del self.players_position[i]

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

            # Append missing values
            for k, v in data.items():
                if len(v) < number_frames:
                    data[k].append(None)

        print("data_analytics: missing values")
        for k, v in data.items():
            print(f"data_analytics: {k} - {len([v for x in v if x is None])}/{len(v)}")
  
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


        



        
    