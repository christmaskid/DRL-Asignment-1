
"""#### Testing"""

import numpy as np
import pickle
import random
from q4_train import get_state

try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
        print("Load.", len(q_table.keys()))
        for key in list(q_table.keys())[:100]:
          print(key)
except FileNotFoundError:
    print("Fail to load q_table.pkl. Use empty q_table instead.")
    q_table = {}


def get_state(obs):
    taxi_pos = obs[:2]
    stations = obs[2:10]
    obstacles = obs[10:14]
    passenger_look = obs[-2]
    destination_look = obs[-1]

    station_directions = []
    idx_dist = []
    for i in range(0, len(stations), 2):
        station = stations[i:i+2]
        rel_station = [station[0]-taxi_pos[0], station[1]-taxi_pos[1]]
        
        def get_dir(x):
            if x == 0:
                return 0
            return x//abs(x)
        station_direction = [get_dir(rel_station[0]), get_dir(rel_station[1])]
        station_directions += station_direction

    return (*obstacles, *station_directions, passenger_look, destination_look)

def get_action(obs):
    state = list(get_state(obs))
    state[-1] = 1 if state[-1] == True else 0
    state = tuple(state)
    print("State", state)
    if state in q_table:
        return np.argmax(q_table[state])
    else:
        # 0: left, 1: right, 2: forward, 3: pickup, 4: dropoff, 5: toggle, 6: done (unused)
        print("Random")
        return random.choice([0, 1, 2, 3, 4, 5])  # Fallback for unseen states
