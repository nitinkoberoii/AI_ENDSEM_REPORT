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

def error_correction_test(patterns, noise_levels):
    N = patterns.shape[1]
    W = train_hopfield(patterns)
    accuracies = []
    for noise_level in noise_levels:
        success_count = 0
        for pattern in patterns:
            noisy_pattern = pattern.copy()
            noise_indices = np.random.choice(len(noisy_pattern), size=int(noise_level * N), replace=False)
            noisy_pattern[noise_indices] *= -1
            recalled_pattern = recall_hopfield(W, noisy_pattern)
            if np.array_equal(recalled_pattern, pattern):
                success_count += 1
        accuracies.append(success_count / len(patterns))
    return accuracies

patterns = np.array([np.random.choice([-1, 1], size=100) for _ in range(10)])
noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
accuracies = error_correction_test(patterns, noise_levels)
for noise, acc in zip(noise_levels, accuracies):
    print(f"Noise Level: {noise * 100:.0f}%, Retrieval Accuracy: {acc * 100:.2f}%")
