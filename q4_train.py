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
from student_agent import get_state


def tabular_q_learning(env, episodes=5000, alpha=0.1, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.99,
            results_save_path="q_table.pkl"):
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
