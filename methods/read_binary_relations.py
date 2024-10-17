import numpy as np
from numpy._typing import NDArray

def read_binary_relations_from_txt(file_path) -> 'dict[str, NDArray]':
    relations = {}
    current_relation = None
    matrix = []
    matrix_size = 15

    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            
            # Skip empty lines
            if not stripped_line:
                continue
            
            # Check if the line indicates a new relation (e.g., R1 -------------------------)
            if stripped_line.startswith('R') and '---' in stripped_line:
                if current_relation and matrix:
                    # Save the previous relation
                    relations[current_relation] = np.array(matrix, dtype=int)
                    matrix = []
                # Extract relation name (e.g., R1)
                current_relation = stripped_line.split()[0]
                # print(f"Detected new relation: {current_relation} at line {line_number}")
                continue
            
            # If it's a matrix line, parse the numbers
            if current_relation:
                # Split the line into numbers, filtering out any empty strings
                numbers = [int(num) for num in stripped_line.split() if num in ('0', '1')]
                
                if len(numbers) != matrix_size:
                    print(f"Warning: Line {line_number} in {current_relation} does not have {matrix_size} elements.")
                
                matrix.append(numbers)
                
                # If we've read enough rows for the matrix, save it
                if len(matrix) == matrix_size:
                    relations[current_relation] = np.array(matrix, dtype=int)
                    # print(f"Completed reading matrix for {current_relation}")
                    matrix = []
                    current_relation = None  # Reset for next relation

    # Handle the last relation if the file doesn't end with a separator
    if current_relation and matrix:
        relations[current_relation] = np.array(matrix, dtype=int)
        # print(f"Completed reading matrix for {current_relation}")

    return relations

# Example usage
if __name__ == "__main__":
    file_path = 'data/Варіант №60.txt'
    relations = read_binary_relations_from_txt(file_path)
    
    # Display the matrices
    for relation_name, matrix in relations.items():
        print(f"\nMatrix for {relation_name}:")
        print(matrix)
