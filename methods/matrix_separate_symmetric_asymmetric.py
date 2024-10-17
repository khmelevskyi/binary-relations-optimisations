import numpy as np

# Function to read the matrix from a text file
def read_matrix_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Process the lines to convert them into a 2D NumPy array
    matrix = [list(map(int, line.split())) for line in lines]
    return np.array(matrix)

# Function to separate symmetric and asymmetric parts
def separate_symmetric_asymmetric(matrix):
    result = np.full(matrix.shape, "0", dtype=object)  # Fill with "0" initially

    n = matrix.shape[0]  # Get the size of the matrix (assuming square)

    for i in range(n):
        for j in range(n):
            if matrix[i][j] == 1:
                if matrix[j][i] == 1:
                    # If both (i, j) and (j, i) are 1, it's symmetric
                    result[i][j] = "I"
                    result[j][i] = "I"
                elif result[j][i] != "I":  # Prevent overwriting symmetric pairs
                    # If (i, j) is 1 and (j, i) is 0, it's asymmetric
                    result[i][j] = "P"
    
    return result

# Function to classify the matrix
def classify_matrix(result):
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

# Main function
def main():
    # Read matrix from file
    matrix = read_matrix_from_file('data/matrix_to_separate_symmetric_asymetrix.txt')
    
    # Separate symmetric and asymmetric parts
    result = separate_symmetric_asymmetric(matrix)

    # Print result matrix
    for row in result:
        print("  ".join(row))
    
    # Classify the matrix
    classify_matrix(result)

# Run the main function
if __name__ == "__main__":
    main()
