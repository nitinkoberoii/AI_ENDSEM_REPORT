import numpy as np

# Constants
GAMMA = 0.9  # Discount factor
EPSILON = 1e-4  # Convergence threshold
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_PROBABILITIES = [0.8, 0.1, 0.1]  # Intended, perpendicular right, perpendicular left

# Grid dimensions
GRID_HEIGHT = 4
GRID_WIDTH = 5

# Initialize grid with rewards
def initialize_grid(reward):
    rewards = np.full((GRID_HEIGHT, GRID_WIDTH), reward, dtype=float)
    rewards[1, 1] = -np.inf  # Obstacle
    rewards[2, 2] = -np.inf  # Obstacle
    rewards[1, 4] = 1.0  # Positive terminal
    rewards[2, 3] = -1.0  # Negative terminal
    return rewards

# Transition model
def get_next_state(x, y, action):
    if action == "UP":
        return max(x - 1, 0), y
    elif action == "DOWN":
        return min(x + 1, GRID_HEIGHT - 1), y
    elif action == "LEFT":
        return x, max(y - 1, 0)
    elif action == "RIGHT":
        return x, min(y + 1, GRID_WIDTH - 1)
    return x, y

# Value iteration function
def value_iteration(reward):
    rewards = initialize_grid(reward)
    values = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=float)
    delta = float("inf")

    while delta > EPSILON:
        delta = 0
        new_values = values.copy()

        for x in range(GRID_HEIGHT):
            for y in range(GRID_WIDTH):
                if rewards[x, y] == -np.inf:
                    continue  # Skip obstacles
                if rewards[x, y] in [1.0, -1.0]:
                    new_values[x, y] = rewards[x, y]
                    continue  # Skip terminal states

                # Evaluate all possible actions
                action_values = []
                for action in ACTIONS:
                    value = 0
                    intended_state = get_next_state(x, y, action)
                    value += ACTION_PROBABILITIES[0] * values[intended_state]

                    # Perpendicular right and left
                    if action in ["UP", "DOWN"]:
                        right_state = get_next_state(x, y, "RIGHT")
                        left_state = get_next_state(x, y, "LEFT")
                    else:
                        right_state = get_next_state(x, y, "UP")
                        left_state = get_next_state(x, y, "DOWN")
                    value += ACTION_PROBABILITIES[1] * values[right_state]
                    value += ACTION_PROBABILITIES[2] * values[left_state]

                    action_values.append(value)

                new_values[x, y] = rewards[x, y] + GAMMA * max(action_values)
                delta = max(delta, abs(new_values[x, y] - values[x, y]))

        values = new_values

    return values

# Solve for each reward scenario
reward_scenarios = [-2, 0.1, 0.02, 1]
for reward in reward_scenarios:
    print(f"Starting calculations for Reward: {reward}...")
    optimal_values = value_iteration(reward)
    print(f"Optimal Value Function for Reward: {reward}")
    print(optimal_values)
    print("-" * 50)
