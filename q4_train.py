import gym
import gym_minigrid
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
from student_agent import my_get_state


def tabular_q_learning(env, episodes=5000, alpha=0.1, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.99,
            results_save_path="q_table.pkl"):
    q_table = dict()

    train_count = dict()

    rewards_per_episode = []
    epsilon = epsilon_start

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

            old_env_info = (env.taxi_pos[:], env.passenger_loc[:], env.destination[:])
            obs, reward, done, _ = env.step(action)
            next_state = my_get_state(obs)
            if next_state not in q_table:
                q_table[next_state] = np.zeros(6)

            shaped_reward = 0
            def get_dist(pos1, pos2):
                x1, y1 = pos1
                x2, y2 = pos2
                return (x1-x2)**2+(y1-y2)**2
            
            if not env.passenger_picked_up: # primary goal: pick up passengerif not has_key and goal_pos is not None: 
                dist = get_dist(old_env_info[0], old_env_info[1])
                next_dist = get_dist(env.taxi_pos, env.passenger_loc)

                if next_dist == 0:
                    if action == 4: # pickup
                        shaped_reward += 100
                else:
                    shaped_reward += (dist - next_dist)*10
                
            elif not done: # secondary goal: find the destination and drop off
                dist = get_dist(old_env_info[0], old_env_info[2])
                next_dist = get_dist(env.taxi_pos, env.destination)

                if next_dist == 0:
                    if action == 5: # drop off
                        shaped_reward += 100
                else:
                    shaped_reward += (dist - next_dist)*10
            
            # reward += shaped_reward
            total_reward += reward
            
            # Bellman equation
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            state = next_state

            # trajectory.append((state, action, reward))
            trajectory.append((action, reward))

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_end, epsilon * decay_rate)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.3f}")
            print(state, q_table[state], env.passenger_loc, env.destination)
        
        if len(trajectory) < 50:
            print(total_reward, len(trajectory), len(q_table), flush=True) # , trajectory
            
        pickle.dump(q_table, open(results_save_path, "wb"))

    return q_table, rewards_per_episode


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from simple_custom_taxi_env import SimpleTaxiEnv
    from complex_custom_taxi_env import ComplexTaxiEnv

    q_table, rewards = tabular_q_learning(
        ComplexTaxiEnv(grid_size=5, max_num_obstacles=5, fuel_limit=5000),
        # SimpleTaxiEnv(fuel_limit=500), 
        episodes=50000, decay_rate=0.9999, #alpha=0.2
    )
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    # plt.show()
    plt.savefig("training_progress.png")

    for key in q_table.keys():
        print(key)
    print(len(q_table.keys()))
