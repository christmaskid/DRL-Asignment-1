import numpy as np
import pickle
import random
import math

try:
    with open("q_table_new4.pkl", "rb") as f:
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

        self.pickedup = 0
        self.dropoff = 0

    def update(self, taxi_pos, stations, passenger_look, destination_look):
        self.stations = stations

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
    print("Taxi", taxi_pos)
    print("stations", stations)
    print("directions", station_directions)

    
    return (    
        *obstacles, 
        *(station_directions[state_memory.target_station_idx]),
        passenger_look, 
        destination_look,
    )

def get_action(obs):

    state = my_get_state(obs)
    print(state_memory)
    print(state)
    
    if state in q_table:
        print(q_table[state])
        action = np.argmax(q_table[state])
    else:
        # 0: left, 1: right, 2: forward, 3: pickup, 4: dropoff, 5: toggle, 6: done (unused)
        print("Random")
        action = random.choice([0, 1, 2, 3, 4, 5])  # Fallback for unseen states
    
    return action
