import numpy as np
import pickle
import random
import math

try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
        print("Load.", len(q_table.keys()))
        for key in list(q_table.keys())[:100]:
          print(key)
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
            f"Taxi pos: {self.taxi_pos}",
            f"Stations: {self.stations}",
            f"Passenger candidates: {self.passenger_candidates}",
            f"Destination candidates: {self.destination_candidates}",
            f"Passenger picked up: {self.passenger_picked_up}",
            "====================\n"
        ])

    def reset(self):
        self.taxi_pos = None
        self.stations = []
        self.passenger_candidates = []
        self.destination_candidates = []
        self.passenger_picked_up = 0
    
    def pickup(self):
        self.passenger_picked_up = 1

    def update(self, taxi_pos, stations,  passenger_look, destination_look):
        if self.taxi_pos is None:
            # Initial state
            self.taxi_pos = taxi_pos
            self.stations = stations

        if passenger_look and len(self.passenger_candidates) != 1:
            new_passenger_candidates = []
            for dir_vec in DIR_VECTORS:
                candidate = (taxi_pos[0]+dir_vec[0], taxi_pos[1]+dir_vec[1])
                if candidate in stations:
                    new_passenger_candidates.append(candidate)
            
            if len(self.passenger_candidates) == 0:
                self.passenger_candidates = new_passenger_candidates
            else:
                # find intersection
                self.passenger_candidates = [
                    cand for cand in self.passenger_candidates 
                    if cand in new_passenger_candidates]
            
            if len(self.passenger_candidates) == 0:
                # passenger is in the taxi!
                self.passenger_picked_up = 1
                self.passenger_candidates = [self.taxi_pos]
        
        if destination_look and len(self.destination_candidates) != 1:
            new_destination_candidates = []
            for dir_vec in DIR_VECTORS:
                candidate = (taxi_pos[0]+dir_vec[0], taxi_pos[1]+dir_vec[1])
                if candidate in stations:
                    new_destination_candidates.append(candidate)
            
            if len(self.destination_candidates) == 0:
                self.destination_candidates = new_destination_candidates
            else:
                # find intersection
                self.destination_candidates = [
                    cand for cand in self.destination_candidates
                    if cand in new_destination_candidates
                ]
        
        if len(self.passenger_candidates) == 1 \
                and self.taxi_pos == self.passenger_candidates[0]:
            self.passenger_picked_up = 0.5 # not sure if picked up

    def to_dropoff(self):
        return len(self.destination_candidates) == 1 \
                and self.taxi_pos == self.destination_candidates[0]


state_memory = StateMemory()
state_memory.reset()

def my_get_state(obs, reset_memory=False):
    taxi_pos = obs[:2]
    stations = [tuple(obs[2+i:2+i+2]) for i in range(0, len(obs[2:10]), 2)]
    obstacles = obs[10:14]
    passenger_look = obs[-2]
    destination_look = obs[-1]

    if reset_memory or set(state_memory.stations) != set(stations):
        # A new environment
        state_memory.reset()
    state_memory.update(taxi_pos, stations,  passenger_look, destination_look)

    def get_dir(x):
        if x == 0: return 0
        return x//abs(x)
    def convert_rel_dir(pos):
        rel_dir = (get_dir(pos[0]-taxi_pos[0]), get_dir(pos[1]-taxi_pos[1]))
        return rel_dir
        # if abs(rel_dir[0]) > abs(rel_dir[1]):
        #     return (rel_dir[0], 0)
        # else:
        #     return (0, rel_dir[1])
    # def convert_rel_dir(pos):
    #     rel_pos = (pos[0]-taxi_pos[0], pos[1]-taxi_pos[1])
    #     if rel_pos==(0,0):
    #         return rel_pos
    #     if rel_pos[1]==0:
    #         return (rel_pos[0]//abs(rel_pos[0]), 0)
    #     if rel_pos[0]==0:
    #         return (0, rel_pos[1]//abs(rel_pos[1]))
    #     gcd = math.gcd(rel_pos[0], rel_pos[1])
    #     return (rel_pos[0]//gcd, rel_pos[1]//gcd)

    station_directions = []
    for station in stations:
        station_directions.append(convert_rel_dir(station))
    station_directions = sorted(station_directions)
    
    def convert_candidates(candidates):
        if len(candidates) == 1:
            return convert_rel_dir(candidates[0])
        return None


    if not state_memory.passenger_picked_up \
                and len(state_memory.passenger_candidates)==1:
        return (    
            *obstacles, 
            *station_directions, 
            passenger_look, 
            destination_look,
            convert_candidates(state_memory.passenger_candidates),
        )
    elif state_memory.passenger_picked_up \
                and len(state_memory.destination_candidates)==1:
        return (    
            *obstacles, 
            *station_directions, 
            passenger_look, 
            destination_look,
            convert_candidates(state_memory.destination_candidates),
        )
    else:
        return (    
            *obstacles, 
            *station_directions, 
            passenger_look, 
            destination_look,
            None
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

"""
DEBUG
    if state[0] in [(i,0) for i in range(4)]:
        return 0 # +(1,0)
    if state[0] in [(i,4) for i in range(1,5)]:
        return 1 # +(-1,0)
    if state[0] in [(4,i) for i in range(4)]:
        return 2 # +(0,1)
    if state[0] in [(0,i) for i in range(1,5)]:
        return 3 # +(0,-1)
    return 0
    # (0,0)->(1,0)->...->(4,0)->(4,1)->...->(4,4)->(3,4)->...->(0,4)->(0,3)
    #      0      0     0     2      2     2     1       1    1     3
"""