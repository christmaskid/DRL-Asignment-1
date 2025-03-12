
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
