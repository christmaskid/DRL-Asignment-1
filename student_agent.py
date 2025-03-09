import numpy as np
import pickle
import random

try:
    with open("q_table_250310.pkl", "rb") as f:
        q_table = pickle.load(f)
except FileNotFoundError:
    print("Fail to load q_table.pkl. Use empty q_table instead.")
    q_table = {}

def get_action(obs):
    if obs in q_table:
        return np.argmax(q_table[obs])
    else:
        # 0: left, 1: right, 2: forward, 3: pickup, 4: dropoff, 5: toggle, 6: done (unused)
        # print("Random")
        return random.choice([0, 1, 2, 3, 4, 5])  # Fallback for unseen states