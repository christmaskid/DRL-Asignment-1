
"""#### Testing"""

import numpy as np
import pickle
import random
from student_agent_tabular_new2 import my_get_state, get_action
# from student_agent_dqn_new2 import my_get_state, get_action

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
        print("action=", action)
        obs, reward, done, _ = env.step(action)
        print('obs=',obs, "reward=", reward)
        total_reward += reward
        taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs
        env.render_env((taxi_row, taxi_col), action=action, step=step_count, fuel=env.current_fuel)

        # print()
        step_count += 1
        # if step_count == 20:
        #   break

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")


if __name__ == "__main__":
    from complex_custom_taxi_env import ComplexTaxiEnv
    run_learned_value(ComplexTaxiEnv(grid_size=10, fuel_limit=5000))
    
    # from simple_custom_taxi_env import SimpleTaxiEnv
    # run_learned_value(SimpleTaxiEnv(fuel_limit=500))
