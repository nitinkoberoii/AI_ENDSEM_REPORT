import math
import random

def simulated_annealing(scrambled_tiles, cost_function, generate_neighbor, initial_temp, cooling_rate, threshold):
    current_state = scrambled_tiles
    current_cost = cost_function(current_state)
    temperature = initial_temp

    while temperature > threshold:
        next_state = generate_neighbor(current_state)
        next_cost = cost_function(next_state)
        delta_e = current_cost - next_cost

        if delta_e > 0:
            current_state = next_state
            current_cost = next_cost
        else:
            acceptance_probability = math.exp(delta_e / temperature)
            if random.random() < acceptance_probability:
                current_state = next_state
                current_cost = next_cost

        temperature *= cooling_rate

    return current_state

def cost_function(state):
    return sum(abs(a - b) for a, b in zip(state, sorted(state)))

def generate_neighbor(state):
    neighbor = state[:]
    i, j = random.sample(range(len(state)), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor

if __name__ == "__main__":
    scrambled_tiles = [3, 1, 4, 2, 5]
    initial_temp = 1000
    cooling_rate = 0.95
    threshold = 0.1
    solution = simulated_annealing(scrambled_tiles, cost_function, generate_neighbor, initial_temp, cooling_rate, threshold)
    print("Final state:", solution)
