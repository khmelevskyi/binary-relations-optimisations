import numpy as np
from numpy._typing import NDArray


def display_str_matrix(matrix: NDArray, relation_name=None):
    if relation_name:
        print(f'__{relation_name}__')

    # Calculate the width for the alignment
    max_val_length = max(len(str(i)) for i in range(1, matrix.shape[0] + 1))
    max_val_length = max(max_val_length, max(len(str(val)) for row in matrix for val in row))

    # Create a format string for proper spacing
    cell_format = f"{{:>{max_val_length}}}"  # Align right with the calculated width

    # Print the header (column numbers)
    header = " " * (max_val_length + 2) + "  ".join(cell_format.format(i) for i in range(1, matrix.shape[0] + 1))
    print(header)
    print()

    # Print each row with the row number
    for i, row in enumerate(matrix, start=1):
        row_str = "  ".join(cell_format.format(val) for val in row)
        print(f"{cell_format.format(i)}  {row_str}")
    print()

# Function to separate symmetric and asymmetric parts
def separate_symmetric_asymmetric(matrix: NDArray):
    # Fill with "N" initially (incomparable, when both (i, j) and (j, i) are 0)
    result = np.full(matrix.shape, "N", dtype=str)

    n = matrix.shape[0]

    for i in range(n):
        for j in range(n):
            if matrix[i][j] == 1:
                if matrix[j][i] == 1:
                    # If both (i, j) and (j, i) are 1, it's symmetric
                    result[i][j] = "I"
                    result[j][i] = "I"
                else:
                    # If (i, j) is 1 and (j, i) is 0, it's asymmetric
                    result[i][j] = "P"
                    result[j][i] = "0"
    
    return result

# Function to classify the matrix
def classify_matrix(result: NDArray):
    n = result.shape[0]
    contains_p = False
    contains_i = False
    only_diagonal_i = True

    for i in range(n):
        for j in range(n):
            if result[i][j] == "P":
                contains_p = True
            if result[i][j] == "I":
                contains_i = True
                if i != j:  # Check if 'I' is outside the diagonal
                    only_diagonal_i = False

    if contains_i and not contains_p:
        print("The matrix is symmetric.")
    elif contains_p and not contains_i:
        print("The matrix is asymmetric.")
    elif contains_p and contains_i:
        if only_diagonal_i:
            print("The matrix is antisymmetric.")
        else:
            print("The matrix is neither symmetric, antisymmetric, nor asymmetric.")


if __name__ == "__main__":
    # Function to read the matrix from a text file
    def read_matrix_from_file(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Process the lines to convert them into a 2D NumPy array
        matrix = [list(map(int, line.split())) for line in lines]
        return np.array(matrix)
    
    file_path = 'data/matrix_to_separate_symmetric_asymetrix.txt'
    matrix = read_matrix_from_file(file_path)
    
    result = separate_symmetric_asymmetric(matrix)

    for row in result:
        print("  ".join(row))

    classify_matrix(result)
    
