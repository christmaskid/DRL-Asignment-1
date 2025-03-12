import gym
import numpy as np
import importlib.util
import time
import random
from IPython.display import clear_output

# This environment allows you to verify whether your program runs correctly during testing,
# as it follows the same observation format from `env.reset()` and `env.step()`.
# However, keep in mind that this is just a simplified environment.
# The full specifications for the real testing environment can be found in the provided spec.
#
# You are free to modify this file to better match the real environment and train your own agent.
# Good luck!

class ComplexTaxiEnv():
    def __init__(self, fuel_limit=5000, grid_size=5, max_num_obstacles=4):
        self.action_size = 6
        self.grid_size = grid_size

        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.max_num_obstacles = max_num_obstacles

        self.stations = []
        self.passenger_loc = None
        self.passenger_picked_up = False
        self.obstacles = set()  # random obstacles
        self.destination = None

        self.max_steps = 256
        self.step_count = 0

    def get_random_positions_exclude(self, occupied, num_return):
        # print(occupied, num_return)
        occupied_map = np.zeros((self.grid_size, self.grid_size))
        for occ in occupied:
          occupied_map[occ[0], occ[1]] = 1
        candidates = []
        for i in range(self.grid_size):
          for j in range(self.grid_size):
            if occupied_map[i][j] == 0:
              candidates.append((i, j))
        random.shuffle(candidates)
        return candidates[:min(num_return, len(candidates))]

    def get_random_positions_from(self, occupied, num_return):
        candidates = occupied[:]
        random.shuffle(candidates)
        return candidates[:min(num_return, len(candidates))]

    def reset(self, **kwargs):
        self.current_fuel = self.fuel_limit
        self.stations = self.get_random_positions_exclude([], 4)
        self.passenger_loc, self.destination = self.get_random_positions_from(self.stations, 2)
        self.taxi_pos = self.get_random_positions_exclude(self.stations, 1)[0]
        taxi_row, taxi_col = self.taxi_pos
        self.passenger_picked_up = False

        # Create obstacles (TODO: and ensure that there's at least a way out)
        num_obstacles = random.randint(0, self.max_num_obstacles)  # Random number of obstacles
        self.obstacles = set(
          self.get_random_positions_exclude(
              self.stations + [self.passenger_loc, self.destination, self.taxi_pos],
              num_obstacles
            )
          )

        self.step_count = 0

        return self.get_state(), {"prob": None, "action_mask": None}

    def collide(self, row, col):
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            # print("hit the border")
            return True
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for direction in directions:
            next_row = row + direction[0]
            next_col = col + direction[1]
            if (next_row, next_col) in self.obstacles:
                # print((next_row, next_col), "Hit obstacle", self.obstacles)
                return True
        return False

    def get_state(self):
        taxi_row, taxi_col = self.taxi_pos
        passenger_x, passenger_y = self.passenger_loc
        destination_x, destination_y = self.destination
        obstacle_north = int(taxi_row == 0 or (taxi_row-1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row+1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col+1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row, taxi_col+1) in self.obstacles)
        passenger_loc_north = int( (taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int( (taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int( (taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int( (taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int( (taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
        destination_loc_north = int( (taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int( (taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int( (taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int( (taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle  = int( (taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle

        state = (taxi_row, taxi_col,
              self.stations[0][0],self.stations[0][1],
              self.stations[1][0],self.stations[1][1],
              self.stations[2][0],self.stations[2][1],
              self.stations[3][0],self.stations[3][1],
              obstacle_north, obstacle_south, obstacle_east, obstacle_west,
              passenger_look, destination_look)
        return state



    def step(self, action):
        """Perform an action and update the environment state."""

        next_row, next_col = self.taxi_pos
        if action == 0 :  # Move South
            next_row += 1
        elif action == 1:  # Move North
            next_row -= 1
        elif action == 2 :  # Move East
            next_col += 1
        elif action == 3 :  # Move West
            next_col -= 1

        terminate = False
        truncated = False
        reward = 0

        if action in [0, 1, 2, 3]:
            self.current_fuel -= 1
            if self.collide(next_row, next_col):
                reward -= 5
                # print("Run into obstacle")
            else:
                reward = -0.1
                self.taxi_pos = next_row, next_col

            if self.current_fuel <= 0:
                reward -= 10
                terminate = True
                # print("Run out of fuel")

        elif action == 4:
            if not self.passenger_picked_up:
                if self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos
                else:
                    reward -= 10
                    # print("incorrect pickup")
            else:
                reward -= 10
                # print("nonsense pickup")

        elif action == 5:
            if self.passenger_picked_up:
                if self.taxi_pos == self.destination:
                  reward += 50
                  terminate = True
                  # print("task completion")
                else:
                  self.passenger_picked_up = False
                  self.passenger_loc = self.taxi_pos
                  # print("incorrect dropoff")
            else:
                reward -= 10
                # print("NONESENSE dropoff")

        if self.passenger_picked_up:
            self.passenger_loc = self.taxi_pos
        if self.step_count >= self.max_steps:
            truncated = True

        return (self.get_state(), reward, terminate, truncated, {})

    def render_env(self, taxi_pos,   action=None, step=None, fuel=None):
        # clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]

        colors = "RGYB"
        for i in range(len(self.stations)):
          grid[self.stations[i][0]][self.stations[i][1]]=colors[i]

        # Place passenger
        py, px = self.passenger_loc
        if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
            grid[py][px] = 'P'

        # Place destination
        dy, dx = self.destination
        if 0 <= dx < self.grid_size and 0 <= dy < self.grid_size:
            grid[dy][dx] = 'D'

        # Plac obstacles
        for obstacle in self.obstacles:
            grid[obstacle[0]][obstacle[1]] = 'X'

        # Place taxi
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = '🚖'


        # Print step info
        print(f"\nStep: {step}")
        print(f"Stations: {self.stations}")
        print(f"Obstacles: {self.obstacles}")
        print(f"Taxi Position: ({tx}, {ty})")
        print(f"Passenger pickup: {self.passenger_picked_up}")
        print(f"Passenger Position: ({px}, {py})")
        print(f"Destination: ({dx}, {dy})")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")
        print(self.get_state())

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"
