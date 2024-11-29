import numpy as np

def train_hopfield(patterns):
    N = patterns.shape[1]
    W = np.zeros((N, N))
    for pattern in patterns:
        W += np.outer(pattern, pattern)
    np.fill_diagonal(W, 0)
    return W / len(patterns)

def recall_hopfield(W, pattern, max_iterations=100):
    current_pattern = pattern.copy()
    for _ in range(max_iterations):
        for i in range(len(current_pattern)):
            current_pattern[i] = 1 if np.dot(W[i], current_pattern) >= 0 else -1
    return current_pattern

def find_capacity(patterns, max_patterns):
    N = patterns.shape[1]
    capacity = 0
    for P in range(1, max_patterns + 1):
        selected_patterns = patterns[:P]
        W = train_hopfield(selected_patterns)
        success = True
        for pattern in selected_patterns:
            noisy_pattern = pattern.copy()
            noise_indices = np.random.choice(len(noisy_pattern), size=int(0.2 * N), replace=False)
            noisy_pattern[noise_indices] *= -1
            recalled_pattern = recall_hopfield(W, noisy_pattern)
            if not np.array_equal(recalled_pattern, pattern):
                success = False
                break
        if not success:
            break
        capacity = P
    return capacity

patterns = np.array([np.random.choice([-1, 1], size=100) for _ in range(20)])
empirical_capacity = find_capacity(patterns, max_patterns=20)
print(f"Empirical capacity of the Hopfield network: {empirical_capacity}")
