import numpy as np

def calculate_energy(x, alpha, beta):
    row_constraints = sum((np.sum(x, axis=1) - 1) ** 2)
    col_constraints = sum((np.sum(x, axis=0) - 1) ** 2)
    return alpha * row_constraints + beta * col_constraints

def solve_eight_rook(alpha=10, beta=10, max_iterations=1000):
    x = np.random.randint(0, 2, size=(8, 8))
    for _ in range(max_iterations):
        energy = calculate_energy(x, alpha, beta)
        for i in range(8):
            for j in range(8):
                x[i, j] = 1 - x[i, j]
                new_energy = calculate_energy(x, alpha, beta)
                if new_energy < energy:
                    energy = new_energy
                else:
                    x[i, j] = 1 - x[i, j]
        if energy == 0:
            break
    return x

solution = solve_eight_rook()
print("Eight Rook Solution:")
print(solution)
