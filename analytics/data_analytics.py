from dataclasses import dataclass
from copy import deepcopy


class InvalidDataPoint(Exception):
    pass


@dataclass
class PlayerPosition:

    id: int
    position: tuple[float, float]

    @property
    def key(self) -> str:
        return f"player{self.id}"

@dataclass
class DataPoint:

    frame: int = None
    players_position: list[PlayerPosition] = None

    def validate(self):
        if self.frame is None:
            raise InvalidDataPoint("Unknown frame")
        
        players_ids = []
        for i, player_pos in enumerate(deepcopy(self.players_position)):
            player_id = player_pos.id

            if player_id in (1, 2, 3, 4):
                players_ids.append(player_id)
            else:
                del self.players_position[i]

        if len(players_ids) != len(set(players_ids)):
            print("HEREEEE ", self.frame)
            raise InvalidDataPoint("N-plicate player id")
        
        if len(self.players_position) != 4:
            # raise InvalidDataPoint
            for id in (1, 2, 3, 4):
                if id not in players_ids:
                    self.add_player_position(
                        PlayerPosition(
                            id=id,
                            position=(None, None),
                        )
                    )
        
        
            
    
    def add_player_position(self, player_position: PlayerPosition):
        if self.players_position is None:
            self.players_position = [player_position]
        else:
            self.players_position.append(player_position)

    def sorted_players_position(self):
        players_position = sorted(
            self.players_position, 
            key=lambda x: x.id,
        )
        return players_position

class DataAnalytics:

    def __init__(self):
        self.frames = [0]
        self.current_datapoint = DataPoint(frame=self.frames[-1])
        self.datapoints: list[DataPoint] = []

    def update(self):
        self.current_datapoint.validate()
        self.datapoints.append(self.current_datapoint)
        self.current_datapoint = DataPoint(frame=self.frames[-1])

    def new_frame(self, frame: int): 

        assert frame not in self.frames

        self.frames.append(frame)
        self.update()
    
    def step(self, x: int = 1) -> None:
        new_frame = self.frames[-1] + 1
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
            players_position = datapoint.sorted_players_position()
            for player_position in players_position:
                data[f"{player_position.key}_x"].append(
                    player_position.position[0]
                )
                data[f"{player_position.key}_y"].append(
                    player_position.position[1],
                )
               
        return data

    