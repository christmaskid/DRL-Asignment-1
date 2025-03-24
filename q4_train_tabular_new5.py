import os
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
import pickle
from tqdm import tqdm

"""### Q-learning (NEW)

#### State
"""

import math
from student_agent_new5 import my_get_state


def tabular_q_learning(env, episodes=5000, alpha=0.1, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.99,
            results_save_path="q_table_new5.pkl"):
    q_table = dict()

    rewards_per_episode = []
    epsilon = epsilon_start
    last_positions = []
    last_actions = []

    for episode in tqdm(range(episodes)):
        obs, _ = env.reset()
        state = my_get_state(obs, reset_memory=True)
        done = False
        total_reward = 0
        trajectory = []

        while not done:
            if state not in q_table:
                q_table[state] = np.zeros(6)

            # ε-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.choice(np.arange(6))
            else:
                action = np.argmax(q_table[state])

            old_passenger_picked_up = int(env.passenger_picked_up)
            obs, reward, done, _ = env.step(action)
            next_state = my_get_state(obs)

            shaped_reward = 0
            
            # obstacles = state[:4]
            target_station_direction = state[4:6]
            DIR_VEC = [
                np.array((1,0)), np.array((-1,0)),
                np.array((0,1)), np.array((0,-1))
            ]
            if action in [0,1,2,3]:
                shaped_reward += np.sum(target_station_direction * DIR_VEC[action]) * 1.5
                
                # Prevent the agent from looping
                for pos in last_positions[-min(len(last_positions), 10):]:
                    if env.taxi_pos == pos:
                        shaped_reward -= 10

                # def get_opposite_dir(a):
                #     opp = [1, 0, 3, 2]
                #     return opp[a]
                # if len(last_actions)>=2 \
                #      and (last_actions[-1] == get_opposite_dir(action) or last_actions[-2] == get_opposite_dir(action)):
                #     shaped_reward -= 10
                # for window_size in range(2, 30):
                #     if len(last_actions) < 2*window_size:
                #         break
                #     if last_actions[-2*window_size:-window_size] == last_actions[-window_size:]:
                #         shaped_reward -= 10 # loop detected

            elif action == 4:
                if not old_passenger_picked_up \
                        and env.taxi_pos == env.passenger_loc:
                    # primary goal: pick up passenger if not has_key and goal_pos is not None: 
                    shaped_reward += 50
                else:
                    shaped_reward -= 50

            elif action == 5:
                if old_passenger_picked_up \
                        and env.taxi_pos == env.destination:
                    # secondary goal: find the destination and drop off
                    shaped_reward += 100
                else:
                    shaped_reward -= 100
            # if done:
            #     shaped_reward += 100
            if reward == -5: # collision
                shaped_reward -= 1000
            if reward == -10:
                shaped_reward -= 500


            last_positions.append(env.taxi_pos)
            last_actions.append(action)
            
            reward += shaped_reward
            total_reward += reward
            
            if next_state not in q_table:
                q_table[next_state] = np.zeros(6)
            # Bellman equation
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            state = next_state

            # trajectory.append((state, action, reward))
            trajectory.append((action, reward))

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_end, epsilon * decay_rate)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.3f}", flush=True)
            # print("\n", len(q_table.keys()), state, q_table[state], env.passenger_loc, env.destination, end="\n", flush=True)

            pickle.dump(q_table, open(results_save_path, "wb"))
        
        # if len(trajectory) < 50:
        #     print(total_reward, len(trajectory), flush=True) # , trajectory
            
        

    return q_table, rewards_per_episode


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from simple_custom_taxi_env import SimpleTaxiEnv
    from complex_custom_taxi_env import ComplexTaxiEnv
    from env import DynamicTaxiEnv

    q_table, rewards = tabular_q_learning(
        # ComplexTaxiEnv(grid_size=12, max_num_obstacles=20, fuel_limit=3000),
        # SimpleTaxiEnv(fuel_limit=500), 
        DynamicTaxiEnv(),
        episodes=100000, decay_rate=0.999, #alpha=0.2
    )
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    # plt.show()
    plt.savefig("training_progress_new5.png")

    # for key in q_table.keys():
    #     print(key)
    # print(len(q_table.keys()))
