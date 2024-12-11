import numpy as np
import itertools

# Function to read the input file
def read_input(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # Dynamically determine the evaluation matrix lines
    evaluation_matrix = []
    line_index = 0
    for line in lines:
        if all(char.isdigit() or char.isspace() for char in line.strip()):  # Check if the line is numeric
            evaluation_matrix.append(list(map(int, line.strip().split())))
            line_index += 1
        else:
            break
    
    evaluation_matrix = np.array(evaluation_matrix)
    
    # Find strict order (V1) and quasi-order (V2) lines
    strict_order = None
    quasi_order = None
    for i, line in enumerate(lines[line_index:], start=line_index):
        if ">" in line and "}" not in line:
            strict_order = line.strip().split(">")
        elif ">" in line and "}" in line:
            quasi_order = line.strip().split("<")
    
    if strict_order is None or quasi_order is None:
        print("Strict order (V1) or quasi-order (V2) not found in the file.")
    
    # Parse quasi-order into groups
    if quasi_order is not None:
        quasi_order = [set(group.strip("{}").split(",")) for group in quasi_order]
    
    return evaluation_matrix, strict_order, quasi_order


# Pareto Relation
def pareto_relation(matrix):
    num_alternatives = matrix.shape[0]
    pareto_matrix = np.zeros((num_alternatives, num_alternatives), dtype=int)

    for i, x in enumerate(matrix):
        for j, y in enumerate(matrix):
            if i != j:
                if np.all(x >= y) and np.any(x > y):  # Pareto dominance
                    pareto_matrix[i, j] = 1

    # Extract Pareto-optimal alternatives
    pareto_optimal_set = {i for i in range(num_alternatives) if not np.any(pareto_matrix[:, i] == 1)}

    return pareto_matrix, pareto_optimal_set

# Majoritarian Rule
def majoritarian_relation(matrix):
    num_alternatives = matrix.shape[0]
    majority_matrix = np.zeros((num_alternatives, num_alternatives), dtype=int)

    for i, x in enumerate(matrix):
        for j, y in enumerate(matrix):
            majority_matrix[i, j] = np.sum(x > y)

    majority_optimal = np.argmax(majority_matrix.sum(axis=1))
    return majority_matrix, {majority_optimal}

# Function to compute lexicographic preference
def lexicographic_preference(evaluation_matrix, strict_order):
    strict_indices = [int(k[1:]) - 1 for k in strict_order]
    sorted_matrix = evaluation_matrix[:, strict_indices]
    sorted_indices = sorted(range(sorted_matrix.shape[0]), key=lambda i: tuple(sorted_matrix[i]), reverse=True)
    return sorted_indices[0]

# Function to compute Berezovsky relation
def berezovsky_relation(evaluation_matrix, quasi_order):
    important_classes = [sorted([int(k[1:]) - 1 for k in group]) for group in quasi_order]
    sorted_matrix = evaluation_matrix[:, list(itertools.chain.from_iterable(important_classes))]
    sorted_indices = sorted(range(sorted_matrix.shape[0]), key=lambda i: tuple(sorted_matrix[i]), reverse=True)
    return sorted_indices[0]

# Function to compute Podinovsky relation
def podinovsky_relation(matrix):
    sorted_vectors = np.sort(matrix, axis=1)[:, ::-1]
    return pareto_relation(sorted_vectors)

# Display results
def display_results(optimal_sets):
    for method, result in optimal_sets.items():
        print(f"{method}: {result}")

# Main function
def main():
    methods = {
        # "Pareto": pareto_relation,
        "Majoritarian": majoritarian_relation,
        # "Lexicographic": lambda mat: lexicographic_relation(mat, strict_order),
        # "Berezovsky": lambda mat: berezovsky_relation(mat, quasi_order),
        # "Podynovsky": podinovsky_relation
    }

    if len(methods) == 1:
        input_file = f"{str(list(methods.keys())[0]).lower()}_test.txt"
    elif len(methods) == 5:
        input_file = "Варіант №60.txt"

    evaluation_matrix, strict_order, quasi_order = read_input(f"input/{input_file}")
    print(evaluation_matrix)
    
    for method, func in methods.items():
        relation_matrix, optimal_set = func(evaluation_matrix)
        print(f"--- {method} Relation ---")
        print("Relation Matrix:")
        print(relation_matrix)
        print("Optimal Alternatives:", optimal_set)
        print()

if __name__ == "__main__":
    main()
