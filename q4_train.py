import gym
import gym_minigrid
import os
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from complex_custom_taxi_env import ComplexTaxiEnv
import pickle
from tqdm import tqdm

"""### Q-learning (NEW)

#### State
"""

import math

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
    #   if rel_station == [0, 0]:
    #     station_direction = [0, 0]
    #   elif rel_station[0] == 0:
    #     station_direction = [0, 1] if rel_station[1] > 0 else [0, -1]
    #   elif rel_station[1] == 0:
    #     station_direction = [1, 0] if rel_station[0] > 0 else [-1, 0]
    #   else:
    #     gcd = math.gcd(rel_station[0], rel_station[1])
    #     # dist = math.sqrt(rel_station[0]**2 + rel_station[1]**2)
    #     # station_direction = [rel_station[0]*1/dist, rel_station[1]*1/dist]
    #     station_direction = [rel_station[0]//gcd, rel_station[1]//gcd]
        def get_dir(x):
            if x == 0:
                return 0
            return x//abs(x)
        station_direction = [get_dir(rel_station[0]), get_dir(rel_station[1])]
        station_directions += station_direction

    return (*obstacles, *station_directions, passenger_look, destination_look)

"""#### Tabular Q-learning"""


def tabular_q_learning(env, episodes=5000, alpha=0.1, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.999,
            results_save_path="q_table.pkl"):
    """
    ✅ Implementing Tabular Q-Learning with Epsilon Decay
    - Uses a **Q-table** to store action values for each state.
    - Updates Q-values using the **Bellman equation**.
    - Implements **ε-greedy exploration** for action selection.
    """
    q_table = dict()

    rewards_per_episode = []
    epsilon = epsilon_start

    for episode in tqdm(range(episodes)):
        obs, _ = env.reset()
        state = get_state(obs)
        done = False
        total_reward = 0
        last_min_dist = None

        trajectory = []
        while not done:
            if state not in q_table:
                q_table[state] = np.zeros(env.action_size) # q_table[state][action] = Q(s,a)

            # ε-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.choice(np.arange(env.action_size))
            else:
                action = np.argmax(q_table[state])

            obs, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            total_reward += reward
            next_state = get_state(obs)
            if next_state not in q_table:
                q_table[next_state] = np.zeros(env.action_size)

            # Bellman equation
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state])) - q_table[state][action]

            state = next_state

            trajectory.append((state, action, reward))

        rewards_per_episode.append(total_reward)
        

        epsilon = max(epsilon_end, epsilon * decay_rate)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.3f}")
            print(total_reward, len(trajectory), len(q_table), flush=True)#, trajectory)

        pickle.dump(q_table, open(results_save_path, "wb"))

    return q_table, rewards_per_episode


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    q_table, rewards = tabular_q_learning(ComplexTaxiEnv(grid_size=10, max_num_obstacles=10),
                                        episodes=20000, decay_rate=0.9999)
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    # plt.show()
    plt.savefig("training_progress.png")

    # for key in q_table.keys():
    #     print(key)
    print(len(q_table.keys()))
