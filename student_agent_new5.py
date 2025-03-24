import numpy as np
import pickle
import random
import math

try:
    with open("q_table_new5.pkl", "rb") as f:
        q_table = pickle.load(f)
        print("Load.", len(q_table.keys()))
        # for key in list(q_table.keys())[:100]:
        #   print(key)
except FileNotFoundError:
    print("Fail to load q_table.pkl. Use empty q_table instead.")
    q_table = {}


DIR_VECTORS = [
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]
]

class StateMemory:
    def __init__(self):
        self.reset()

    def __repr__(self):
        return "\n".join([
            "===[State memory]===",
            f"Target station idx: {self.target_station_idx}",
            f"Target station: {self.stations[self.target_station_idx]}",
            f"Look: {self.pickedup, self.dropoff}",
            "====================\n"
        ])

    def reset(self):
        self.stations = []
        self.target_station_idx = 0
        # self.last_pos = (0, 0)
        # self.last_last_pos = (0, 0)
        self.window_size = 3
        self.last_positions = [(None, None)] * self.window_size

        self.pickedup = 0
        self.dropoff = 0

    def update(self, taxi_pos, stations, passenger_look, destination_look):
        self.stations = stations
        # self.last_last_pos = self.last_pos
        # self.last_pos = taxi_pos
        for i in range(self.window_size-1):
            self.last_positions[i] = self.last_positions[i+1]
        self.last_positions[-1] = taxi_pos

        if taxi_pos == stations[self.target_station_idx]:
            
            if passenger_look and not destination_look:
                if self.pickedup:
                    self.target_station_idx = (self.target_station_idx+1)%4
                else:
                    self.pickedup = 1

            elif passenger_look and destination_look:
                if self.dropoff:
                    self.target_station_idx = (self.target_station_idx+1)%4
                else:
                    self.dropoff = 1
            
            else:
                self.target_station_idx = (self.target_station_idx+1)%4
                self.pickedup = 0
                self.dropoff = 0


state_memory = StateMemory()
state_memory.reset()

def my_get_state(obs, reset_memory=False):
    taxi_pos = obs[:2]
    stations = sorted(tuple(obs[2+i:2+i+2]) for i in range(0, len(obs[2:10]), 2))
    obstacles = obs[10:14]
    passenger_look = obs[-2]
    destination_look = obs[-1]

    if reset_memory:
        # A new environment
        state_memory.reset()
    # last_pos = state_memory.last_pos
    # last_last_pos = state_memory.last_last_pos
    last_positions = state_memory.last_positions
    state_memory.update(taxi_pos, stations, passenger_look, destination_look)

    def get_dir(x):
        if x == 0: return 0
        return x//abs(x)
    def convert_rel_dir(pos):
        rel_dir = (get_dir(pos[0]-taxi_pos[0]), get_dir(pos[1]-taxi_pos[1]))
        return rel_dir
    # def convert_rel_pos(pos):
    #     rel_pos = (pos[0]-taxi_pos[0], pos[1]-taxi_pos[1])
    #     return rel_pos

    station_directions = []
    for station in stations:
        station_directions.append(convert_rel_dir(station))
    # print("Taxi", taxi_pos)
    # print("stations", stations)
    # print("directions", station_directions)

    last_positions_rel = []
    for pos in last_positions:
        if pos == (None, None):
            last_positions_rel.append((0, 0))
        else:
            last_positions_rel.append((pos[0]-taxi_pos[0], pos[1]-taxi_pos[1]))

    return (    
        *obstacles, 
        *(station_directions[state_memory.target_station_idx]),
        passenger_look, 
        destination_look,
        int(taxi_pos in stations),
        # tuple(np.array(last_last_pos) - np.array(taxi_pos)),
        # tuple(np.array(last_pos) - np.array(taxi_pos))
        *last_positions_rel
    )

def get_action(obs):

    state = my_get_state(obs)
    # print(state_memory)
    # print(state)
    
    if state in q_table:
        # print(q_table[state])
        action = np.argmax(q_table[state])
        # print(f"{action}({state[4:6]},{state_memory.target_station_idx})", end=" ")
    else:
        # 0: left, 1: right, 2: forward, 3: pickup, 4: dropoff, 5: toggle, 6: done (unused)
        # print("Random")
        # Rule-based
        action = random.choice([0, 1, 2, 3])  # Fallback for unseen states
        stations = sorted(tuple(obs[2+i:2+i+2]) for i in range(0, len(obs[2:10]), 2))
        taxi_pos = obs[:2]
        if taxi_pos in stations:
            if state[-2] and not state[-1]:
                action = 4
            elif state[-2] and state[-1]:
                action = 5
        # print(f"{action}(r)", end=" ")
    print(action, end=" ")
    
    return action
