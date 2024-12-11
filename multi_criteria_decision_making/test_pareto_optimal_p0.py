import numpy as np
from multi_criteria_decision_making.main import (
    read_input,
    get_criteria_diffs,
    get_sigma_matrix,
    get_P0_pareto_relation
)

input_file = "Варіант №60.txt"
evaluation_matrix, strict_order, quasi_order = read_input(f"multi_criteria_decision_making/input/{input_file}")

diff_matrix = get_criteria_diffs(evaluation_matrix) # матриця різниць оцінок за критеріями
sigma_matrix = get_sigma_matrix(diff_matrix) # сігма матриця

n_alternatives = sigma_matrix.shape[0]  # Кількість альтернатив
n_criterias = evaluation_matrix.shape[1] # Кількість критерій

pareto_matrix = get_P0_pareto_relation(sigma_matrix, n_alternatives)

# Extract Pareto-optimal alternatives
pareto_optimal_set = sorted(i+1 for i in range(n_alternatives) if not np.any(pareto_matrix[:, i] == 1))

print(f"\n--- Pareto Relation P0 ---")
print("Relation Matrix:")
print(pareto_matrix)
print("Optimal Alternatives:", pareto_optimal_set)
print()