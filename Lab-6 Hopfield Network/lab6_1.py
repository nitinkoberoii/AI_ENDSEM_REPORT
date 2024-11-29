import numpy as np
import matplotlib.pyplot as plt

def train_hopfield(patterns):
    N = patterns.shape[1]  # Number of neurons
    W = np.zeros((N, N))
    for p in patterns:
        W += np.outer(p, p)
    np.fill_diagonal(W, 0)  # No self-connections
    return W / patterns.shape[0]

def recall_hopfield(W, noisy_pattern, max_iterations=100):
    state = noisy_pattern.copy()
    for _ in range(max_iterations):
        new_state = np.sign(W @ state)
        new_state[new_state == 0] = -1
        if np.array_equal(new_state, state):
            break
        state = new_state
    return state

# Example patterns (10x10 binary grid, flattened)
patterns = np.array([
    np.random.choice([-1, 1], size=100),
    np.random.choice([-1, 1], size=100),
])

# Train the network
W = train_hopfield(patterns)

# Create a noisy version of the first pattern
noisy_pattern = patterns[0].copy()
noise_indices = np.random.choice(len(noisy_pattern), size=20, replace=False)
noisy_pattern[noise_indices] *= -1

# Recall the original pattern
recalled_pattern = recall_hopfield(W, noisy_pattern)

# Reshape for visualization
original = patterns[0].reshape(10, 10)
noisy = noisy_pattern.reshape(10, 10)
recalled = recalled_pattern.reshape(10, 10)

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Original Pattern")
plt.imshow(original, cmap="gray")

plt.subplot(1, 3, 2)
plt.title("Noisy Pattern")
plt.imshow(noisy, cmap="gray")

plt.subplot(1, 3, 3)
plt.title("Recalled Pattern")
plt.imshow(recalled, cmap="gray")
plt.show()
