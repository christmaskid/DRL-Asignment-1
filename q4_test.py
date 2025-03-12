
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

def run_learned_value(env, max_steps=100, gif_path="taxiv3_q_learning.gif"):
    total_reward = 0
    done = False
    step_count = 0

    obs, _ = env.reset()
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    env.render_env((taxi_row, taxi_col), action=None, step=step_count, fuel=env.current_fuel)

    step_count = 0
    while not done:
        action = get_action(obs)
        obs, reward, done, _, _ = env.step(action)
        print('obs=',obs, "reward=", reward)
        total_reward += reward
        taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs
        env.render_env((taxi_row, taxi_col), action=action, step=step_count, fuel=env.current_fuel)

        # print()
        step_count += 1
        # if step_count == 10:
        #   break

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")


if __name__ == "__main__":
    from complex_custom_taxi_env import ComplexTaxiEnv
    run_learned_value(ComplexTaxiEnv())
