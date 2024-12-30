import math
import numpy as np
from typing import TextIO
import matplotlib.pyplot as plt

from optimisation_methods.neumann_morgenstern import Neumann_Morgenstern_optimization
from optimisation_methods.acyclicity_check import adj_matrix_to_adj_list, is_acyclic_dfs
from optimisation_methods.matrix_separate_symmetric_asymmetric import classify_matrix, separate_symmetric_asymmetric


# Клас для зручного зберігання та користування вхідними даними
class InputData:
    def __init__(self,
                alts_evals: np.ndarray,
                weights: np.ndarray,
                c_min: float,
                d_max: float):
        self.alts_evals = alts_evals # матриця оцінок альтернатив за критеріями
        self.weights = np.array(weights) # вагові коефіцієнти
        self.c_min = c_min # мінімальне порогове значення узгодженності
        self.d_max = d_max # максимальне порогове значення неузгодженності


# Зчитуємо дані з файлу
def read_input(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    print(lines)
    # Reading the evaluation matrix
    evaluation_matrix = []
    line_index = 0
    for line in lines:
        if all(char.isdigit() or char.isspace() for char in line.strip()):  # Only numbers and spaces
            evaluation_matrix.append(list(map(int, line.strip().split())))
            line_index += 1
        elif line_index == 0:
            continue
        else:
            break

    evaluation_matrix = np.array(evaluation_matrix)

    # Читання вагових коефіцієнтів
    weight_coefficients = []
    for i, line in enumerate(lines[line_index:], start=line_index):
        if "Вагові коефіцієнти" in line:
            weight_coefficients = list(map(int, lines[i + 1].strip().split()))
            line_index = i + 2
            break

    # Читання порогових значень
    thresholds = []
    for i, line in enumerate(lines[line_index:], start=line_index):
        if "порогів для індексів" in line:
            thresholds = list(map(float, lines[i + 1].strip().split()))
            break
    
    if not weight_coefficients or len(weight_coefficients) != evaluation_matrix.shape[1]:
        print("Error: Weighted Coefficients are not found or incomplete.")

    if not thresholds or len(thresholds) != 2:
        print("Error: Concordance and discordance thresholds are not found or incomplete.")

    if thresholds:
        concordance_threshold, discordance_threshold = thresholds
    else:
        concordance_threshold, discordance_threshold = None, None

    # Returning all the extracted components
    return InputData(evaluation_matrix,
                     weight_coefficients,
                     concordance_threshold,
                     discordance_threshold)


### GET ELECTRE ONE RESULT ###
def write_binary_matrix_to_file(name: str, matrix: np.ndarray, file: TextIO):
    file.write(f"{name}\n")
    file.write("\n".join(list(map(lambda row: f" {'  '.join(map(lambda e: str(int(e)), row))} ", matrix))))
    file.write("\n")

def write_float_matrix_to_file(name: str, matrix: np.ndarray, file: TextIO):
    file.write(f"{name}\n")
    file.write("\n".join(list(map(lambda row: f" {' '.join(map(lambda e: f'{e:.3f}', row))} ", matrix))))
    file.write("\n")


def calculate_concordance_matrix(alternatives_evaluations: np.ndarray, weights: np.ndarray):
    size = len(alternatives_evaluations)
    weights_sum = weights.sum()
    concordance_matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            concordance_matrix[i, j] = sum(
                weights[k] for k in range(len(weights))
                if alternatives_evaluations[i, k] >= alternatives_evaluations[j, k]
            ) / weights_sum

    return concordance_matrix


def calculate_discordance_matrix(alternatives_evaluations: np.ndarray, weights: np.ndarray):
    size = len(alternatives_evaluations)
    sigmas = alternatives_evaluations.max(axis=0) - alternatives_evaluations.min(axis=0)
    weights_multiplied_by_sigmas = weights * sigmas
    discordance_matrix = np.identity(size)

    for i in range(size):
        for j in range(size):
            if i == j:
                continue

            max_weight_by_b_sub_a = -math.inf
            max_weight_by_sigma = -math.inf

            for k in range(len(weights)):
                if alternatives_evaluations[i, k] >= alternatives_evaluations[j, k]:
                    continue

                current_weight_by_b_sub_a = (
                    weights[k] * (alternatives_evaluations[j, k] - alternatives_evaluations[i, k])
                )
                max_weight_by_b_sub_a = max(max_weight_by_b_sub_a, current_weight_by_b_sub_a)

                current_weight_by_sigma = weights_multiplied_by_sigmas[k]
                max_weight_by_sigma = max(max_weight_by_sigma, current_weight_by_sigma)

            discordance_matrix[i, j] = max_weight_by_b_sub_a / max_weight_by_sigma

    return discordance_matrix


def calculate_outranking_relation(c_matrix: np.ndarray, d_matrix: np.ndarray, c_min: float, d_max: float):
    size = len(c_matrix)
    outranking_relation = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i != j and c_matrix[i, j] >= c_min and d_matrix[i, j] <= d_max:
                outranking_relation[i, j] = 1

    return outranking_relation


def get_outranking_relation_kernel(outranking_relation: np.ndarray):
    R_matrix = outranking_relation
    adj_list = adj_matrix_to_adj_list(adj_matrix=R_matrix) # get outgoing sets

    result = separate_symmetric_asymmetric(R_matrix)
    classify_matrix(result)
    
    if is_acyclic_dfs(adj_list):
        print(f"БВ є ациклічним. Використовуємо метод Неймана-Моргенштерна\n")
        solution = Neumann_Morgenstern_optimization(R_matrix)
        print(f"\n________")
        print(f"Xнм: {solution}\n")
        return solution

    return None


def electre_one(data: InputData):
    c_matrix = calculate_concordance_matrix(data.alts_evals, data.weights)
    d_matrix = calculate_discordance_matrix(data.alts_evals, data.weights)
    outranking_relation = calculate_outranking_relation(c_matrix, d_matrix, data.c_min, data.d_max)
    return get_outranking_relation_kernel(outranking_relation)


def electre_one_with_logs(data: InputData, output_file: TextIO):
    c_matrix = calculate_concordance_matrix(data.alts_evals, data.weights)
    write_float_matrix_to_file("Matrix of concordance indices C", c_matrix, output_file)

    d_matrix = calculate_discordance_matrix(data.alts_evals, data.weights)
    write_float_matrix_to_file("Matrix of discordance indices D", d_matrix, output_file)

    output_file.write("Threshold values for concordance and discordance indices c, d\n")
    output_file.write(f"{data.c_min} {data.d_max}\n")

    outranking_relation = calculate_outranking_relation(c_matrix, d_matrix, data.c_min, data.d_max)
    write_binary_matrix_to_file("Outranking relation for thresholds c, d:", outranking_relation, output_file)

    return get_outranking_relation_kernel(outranking_relation)


def get_electre_results_with_logs(data: InputData):
    output_file = open("multi_criteria_dm_electre_one/output/output.txt", "w")
    result = electre_one_with_logs(data, output_file)
    output_file.write("Ядро відношення:\n")
    output_file.write(" ".join(map(str, sorted(result))))
    output_file.write("\n")
    print("Kernel of the outranking relation:")
    print(sorted(result))
    print()
    output_file.close()

### /GET ELECTRE ONE RESULT ###


### EXPERIMENTS PART ###
def print_table_row(element: str, is_acyclic: bool, result_set: set, elements_count: int):
    print_table_row_string(
        element,
        "+" if is_acyclic else "-",
        "-" if result_set is None else str(sorted(result_set)),
        "-" if elements_count is None else elements_count)


def print_table_row_string(element: str, is_acyclic: str, result_set: str, elements_count: str):
    print(f"{element: <10}| {is_acyclic: <10}| {result_set: <60}| {elements_count: <15}")


def execute_d_influence_experiment(data: InputData):
    data_for_exp = InputData(data.alts_evals, data.weights, data.c_min, data.d_max)

    data_for_exp.c_min = 0.754
    step = 0.05
    print("Determination of the effect of changing the threshold value d on the composition and size of the kernel:")
    print_table_row_string("d max", "acyclic", "kernel", "elements count")
    d_maxes = list()
    kernel_sizes = list()
    for d_max in np.arange(0 + step, 0.5+step, step):
        data_for_exp.d_max = d_max
        result = electre_one(data_for_exp)
        print_table_row(f"{d_max:.2f}", result is not None, result, None if result is None else len(result))
        d_maxes.append(d_max)
        kernel_sizes.append(0 if result is None else len(result))

    plt.plot(d_maxes, kernel_sizes)
    plt.title("d threshold influence on the size of the kernel")
    plt.xlabel('d threshold')
    plt.ylabel('kernel size')
    plt.show()


def execute_c_influence_experiment(data: InputData):
    data_for_exp = InputData(data.alts_evals, data.weights, data.c_min, data.d_max)

    data_for_exp.d_max = 0.481
    step = 0.05
    print("Determination of the effect of changing the threshold value c on the composition and size of the kernel:")
    print_table_row_string("c min", "acyclic", "kernel", "elements count")
    c_min_list = list()
    kernel_sizes = list()
    for c_min in np.arange(0.5, 1 + step, step):
        data_for_exp.c_min = c_min
        result = electre_one(data_for_exp)
        print_table_row(f"{c_min:.2f}", result is not None, result, None if result is None else len(result))
        c_min_list.append(c_min)
        kernel_sizes.append(0 if result is None else len(result))

    plt.plot(c_min_list, kernel_sizes)
    plt.title("c threshold influence on the size of the kernel")
    plt.xlabel('c threshold')
    plt.ylabel('kernel size')
    plt.show()


def execute_c_and_d_influence_experiment(data: InputData):
    data_for_exp = InputData(data.alts_evals, data.weights, data.c_min, data.d_max)

    step = 0.05
    data_for_exp.c_min, data_for_exp.d_max = 1.0 + step, 0.0
    print("Determination of the effect of changing the threshold value d on the composition and size of the kernel:")
    print_table_row_string("c min, d max", "acyclic", "kernel", "elements count")
    thresholds = list()
    kernel_sizes = list()
    for _ in np.arange(0, 0.5 + step, step):
        data_for_exp.c_min -= step
        data_for_exp.d_max += step
        result = electre_one(data_for_exp)
        print_table_row(
            f"({data_for_exp.c_min:.2f}, {data_for_exp.d_max:.2f})",
            result is not None,
            result,
            None if result is None else len(result))
        thresholds.append((data_for_exp.c_min, data_for_exp.d_max))
        kernel_sizes.append(0 if result is None else len(result))

    c_min_list = [x for (x, _) in thresholds]
    d_max_list = [y for (_, y) in thresholds]
    plt.plot(c_min_list, d_max_list)
    plt.xlabel('c threshold')
    plt.ylabel('d threshold')
    for i in range(0, len(thresholds), 10):
        plt.text(thresholds[i][0], thresholds[i][1], f"{kernel_sizes[i]} items")
    plt.show()

    plt.plot(
        list(map(lambda thresholds_tuple: f"({thresholds_tuple[0]:.2f}, {thresholds_tuple[1]:.2f})", thresholds)),
        kernel_sizes)
    plt.title("c and d thresholds influence on the size of the kernel")
    plt.xlabel('(c, d) threshold')
    plt.ylabel('kernel size')
    visible_thresholds = thresholds[::10]
    plt.xticks(
        np.arange(0, len(thresholds), 10),
        list(
            map(lambda thresholds_tuple: f"({thresholds_tuple[0]:.2f}, {thresholds_tuple[1]:.2f})",
                visible_thresholds)))
    plt.show()

### /EXPERIMENTS PART ###


### Головна функція ###
def main():
    # Зчитуємо вхідні дані та зберігаємо їх
    input_file = "Варіант №60 умова.txt"

    input_data = read_input(f"multi_criteria_dm_electre_one/input/{input_file}")
    print("\nМатриця оцінок за критеріями множини K:")
    print(input_data.alts_evals)
    print()
    print(input_data.weights)
    print(input_data.c_min, input_data.d_max)


    # Визначаємо ядро методом ELECTRE I
    get_electre_results_with_logs(data=input_data)


    experiments = [
        lambda: execute_d_influence_experiment(input_data),
        lambda: execute_c_influence_experiment(input_data),
        lambda: execute_c_and_d_influence_experiment(input_data)
    ]

    for i in range(len(experiments)):
        print(f"\n\nExperiment {i + 1}")
        experiments[i]()

### Головна функція ###

if __name__ == "__main__":
    main()