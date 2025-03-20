import os
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
import pickle
from tqdm import tqdm
import random


import math
from student_agent_tabular_new2 import my_get_state


def tabular_q_learning(episodes=5000, alpha=0.1, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.99,
            results_save_path="q_table_new4.pkl"):
    q_table = dict()

    train_count = dict()

    rewards_per_episode = []
    epsilon = epsilon_start

    for episode in tqdm(range(episodes)):
        grid_size = random.randint(5, 12)
        max_num_obstacles = int((grid_size**2)/10+1)
        env = ComplexTaxiEnv(grid_size=grid_size, 
                             max_num_obstacles=max_num_obstacles, 
                             fuel_limit=5000)

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
            should_pickup = not old_passenger_picked_up \
                        and env.taxi_pos == env.passenger_loc
            should_dropoff = old_passenger_picked_up \
                        and env.taxi_pos == env.destination
            
            if action in [0,1,2,3]:
                # if should_pickup or should_dropoff:
                #     shaped_reward -= 1
                # else:
                shaped_reward += np.sum(target_station_direction * DIR_VEC[action]) * 1
                
            if action == 4:
                if should_pickup:
                    # primary goal: pick up passengerif not has_key and goal_pos is not None: 
                    shaped_reward += 50
                else:
                    shaped_reward -= 8

            elif action == 5:
                if should_dropoff:
                    # secondary goal: find the destination and drop off
                    shaped_reward += 100
                else:
                    shaped_reward -= 12
            
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
            print("\n", len(q_table.keys()), state, q_table[state], env.passenger_loc, env.destination, end="\n", flush=True)

            pickle.dump(q_table, open(results_save_path, "wb"))
        
        if len(trajectory) < 50:
            print(total_reward, len(trajectory), flush=True) # , trajectory
            
        

    return q_table, rewards_per_episode


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from simple_custom_taxi_env import SimpleTaxiEnv
    from complex_custom_taxi_env import ComplexTaxiEnv

    q_table, rewards = tabular_q_learning(
        # SimpleTaxiEnv(fuel_limit=500), 
        episodes=20000, decay_rate=0.9995, #alpha=0.2
    )
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    # plt.show()
    plt.savefig("training_progress.png")

    # for key in q_table.keys():
    #     print(key)
    # print(len(q_table.keys()))
