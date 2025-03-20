import matplotlib.pyplot as plt
from simple_custom_taxi_env import SimpleTaxiEnv
from complex_custom_taxi_env import ComplexTaxiEnv
from dqn import *
from student_agent_dqn_new2 import my_get_state
import os

import torch
import random
import numpy as np

def set_seed(seed=531):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_dqn(env, hidden_size=64, episodes=5000, alpha=0.1, gamma=0.9, batch_size=64,
              epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.9995, load_path="target_dqn.pth"):
    set_seed(0)
    
    obs, _ = env.reset()
    state_size = len(my_get_state(obs, True)) #
    action_size = 6 # env.action_size
    print("State size", state_size, "action size", action_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = DQNTrainer(state_size, action_size, hidden_size=hidden_size,
        lr=alpha, gamma=gamma, batch_size=batch_size, device=device)
    dqn = trainer.dqn
    target_dqn = trainer.target_dqn

    epsilon = epsilon_start
    rewards_per_episode = []

    trained_states = set() # debug

    for episode in tqdm(range(episodes)):
        obs, _ = env.reset()
        state = my_get_state(obs, True)
        prev_state = state
        done = False
        total_reward = 0

        while not done:

            if state not in trained_states:
                trained_states.add(state)

            action = trainer.get_action(state, epsilon)
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
                shaped_reward += np.sum(target_station_direction * DIR_VEC[action]) * 0.5
                
            if not old_passenger_picked_up \
                    and env.taxi_pos == env.passenger_loc \
                    and action == 4:
                # primary goal: pick up passengerif not has_key and goal_pos is not None: 
                shaped_reward += 50

            elif old_passenger_picked_up \
                    and env.taxi_pos == env.destination \
                    and action == 5:
                # secondary goal: find the destination and drop off
                shaped_reward += 100

            reward += shaped_reward
            total_reward += reward
            trainer.update(state, action, reward, next_state, done)
            prev_state = state
            state = next_state

        epsilon = max(epsilon_end, epsilon * decay_rate)
        rewards_per_episode.append(total_reward)

        if (episode+1) % 50 == 0:
            target_dqn.load_state_dict(dqn.state_dict())
            torch.save(target_dqn.state_dict(), "target_dqn_new2.pth")

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Average Total Reward: {np.mean(rewards_per_episode[-100:])}, Epsilon: {epsilon:.3f}, Explored state: {len(trained_states)}   ")

    return dqn, rewards_per_episode



if __name__=="__main__":
    _, rewards = train_dqn(
        ComplexTaxiEnv(grid_size=10, max_num_obstacles=10, fuel_limit=5000),
        batch_size=64, episodes=5000, decay_rate=0.999,
        hidden_size=128, alpha=0.05
    )
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    # plt.show()
    plt.savefig("training_progress_dqn.png")
