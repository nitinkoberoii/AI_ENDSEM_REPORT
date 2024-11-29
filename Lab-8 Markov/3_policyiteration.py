import numpy as np

# Constants
r = 10  # Revenue per bike rented
t = 2   # Cost per bike moved (except the free one)
parking_cost = 4  # Additional cost if more than 10 bikes are parked
shuttle_cost = 0  # Cost for moving one bike for free (shuttle)
max_bikes = 20  # Maximum bikes allowed at each location
max_move = 5  # Maximum bikes that can be moved overnight
gam = 0.9  # Discount factor

# Poisson distributions (mean requests and returns)
lamda1 = 3  # Expected rental requests at location 1
lamda2 = 4  # Expected rental requests at location 2
mu1 = 3  # Expected returns at location 1
mu2 = 2  # Expected returns at location 2

# State space: number of bikes at each location (0 to 20)
num_states = (max_bikes + 1) * (max_bikes + 1)

# Function to calculate the Poisson probability
def poisson(lamda, k):
    return (lamda ** k) * np.exp(-lamda) / np.math.factorial(k)

# Reward function considering the new costs
def reward_function(state, action):
    m, n = state  # m = bikes at location 1, n = bikes at location 2
    move = action

    # Revenue from rentals (if bikes are available)
    rent1 = min(m, lamda1) * r  # Rent bikes at location 1
    rent2 = min(n, lamda2) * r  # Rent bikes at location 2

    # Cost for moving bikes (shuttle for free, others cost INR 2)
    move_cost = 0
    if move > 0:
        move_cost = (move - 1) * t if move > 1 else 0
    elif move < 0:
        move_cost = (-move) * t

    # Parking cost (if bikes exceed 10)
    parking1_cost = parking_cost if m + move > 10 else 0
    parking2_cost = parking_cost if n - move > 10 else 0

    # Total cost and reward
    total_cost = move_cost + parking1_cost + parking2_cost
    total_reward = rent1 + rent2 - total_cost

    return total_reward

# Transition function (next state given current state and action)
def transition(state, action):
    m, n = state  # m = bikes at location 1, n = bikes at location 2
    move = action

    # Calculate next state after moving bikes
    new_m = min(max(0, m - lamda1 + mu1 + move), max_bikes)
    new_n = min(max(0, n - lamda2 + mu2 - move), max_bikes)

    return new_m, new_n

# Policy iteration function
def policy_iteration():
    # Initialize value function and policy
    V = np.zeros((max_bikes + 1, max_bikes + 1))
    policy = np.zeros((max_bikes + 1, max_bikes + 1), dtype=int)

    # Value iteration loop
    while True:
        delta = 0
        for m in range(max_bikes + 1):
            for n in range(max_bikes + 1):
                # For each state, evaluate the value for each action
                action_values = []
                for move in range(-max_move, max_move + 1):
                    if 0 <= m - move <= max_bikes and 0 <= n + move <= max_bikes:  # valid move
                        next_state = transition((m, n), move)
                        reward = reward_function((m, n), move)
                        action_values.append(reward + gam * V[next_state])
                    else:
                        action_values.append(-np.inf)  # Invalid move, very low value

                # Update the policy and value function
                best_action_value = max(action_values)
                best_action = np.argmax(action_values)
                V[m, n] = best_action_value
                policy[m, n] = best_action

        # Check for convergence (small change in value function)
        if delta < 1e-4:
            break

    return policy, V

# Test the policy iteration function
policy, V = policy_iteration()

# Output the optimal policy and value function
print("Optimal Policy (Number of bikes to move):")
print(policy)
print("Value Function (Expected reward for each state):")
print(V)
