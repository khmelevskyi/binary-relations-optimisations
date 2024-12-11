import re
import numpy as np
from optimisation_methods.main import find_optimal_solution


# Читання вхідного файлу
def read_input(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # Знаходження рядків з файлу на яких знаходиться матриця оцінок альтернатив за критеріями множини K
    evaluation_matrix = []
    line_index = 0
    for line in lines:
        if all(char.isdigit() or char.isspace() for char in line.strip()):  # Перевіряємо ци тільки цифри на рядку
            evaluation_matrix.append(list(map(int, line.strip().split())))
            line_index += 1
        else:
            break
    
    evaluation_matrix = np.array(evaluation_matrix)
    
    # Зчитування строгого порядку (V1) та квазі-порядку (V2) з файлу
    strict_order = None
    quasi_order = None
    for i, line in enumerate(lines[line_index:], start=line_index):
        if ">" in line and "}" not in line:
            strict_order = line.strip().split(">")
        elif "<" in line and "}" in line:
            quasi_order = line.strip().split("<")
    
    if strict_order is None or quasi_order is None:
        print("Strict order (V1) or quasi-order (V2) not found in the file.")

    if strict_order is not None:
        strict_order = [int(re.search(r'\d+', item).group())-1 for item in strict_order]
    
    # Розбиття квазі-порядку на групи
    if quasi_order is not None:
        quasi_order = [list(group.strip("{}").split(",")) for group in quasi_order]
        quasi_order = [[int(re.search(r'\d+', item).group())-1 for item in group] for group in quasi_order]
    
    return evaluation_matrix, strict_order, quasi_order

# Матриця різниць оцінок за критеріями
def get_criteria_diffs(matrix):
    # Calculate the difference matrix
    diff_matrix = np.array([
        [matrix[i] - matrix[j] for j in range(matrix.shape[0])]
        for i in range(matrix.shape[0])
    ])
    # print(diff_matrix)
    return diff_matrix

def get_sigma_matrix(diff_matrix):
    sigma_matrix = np.where(diff_matrix == 0, 0, np.where(diff_matrix > 0, 1, -1))
    # print(sigma_matrix)
    return sigma_matrix


# Відношення Парето R0
def get_R0_pareto_relation(sigma_matrix, n_alternatives):
    # Створення відношення Парето
    pareto_relation = np.zeros((n_alternatives, n_alternatives), dtype=int)
    pareto_relation = np.array([
        [0 if -1 in sigma_matrix[i, j] else 1 for j in range(sigma_matrix.shape[1])]
        for i in range(sigma_matrix.shape[0])
    ])
    return pareto_relation

# Відношення Парето I0 (симетрична частина)
def get_I0_pareto_relation(sigma_matrix, n_alternatives):
    # Створення відношення Парето
    pareto_relation = np.zeros((n_alternatives, n_alternatives), dtype=int)
    pareto_relation = np.array([
        [1 if np.all(sigma_matrix[i, j] == 0) else 0 for j in range(sigma_matrix.shape[1])]
        for i in range(sigma_matrix.shape[0])
    ])
    return pareto_relation

# Відношення Парето N0 (непорівнювальна частина)
def get_N0_pareto_relation(sigma_matrix, n_alternatives):
    # Створення відношення Парето
    pareto_relation = np.zeros((n_alternatives, n_alternatives), dtype=int)
    pareto_relation = np.array([
        [1 if np.any(sigma_matrix[i, j] == 1) and np.any(sigma_matrix[i, j] == -1) else 0 for j in range(sigma_matrix.shape[1])]
        for i in range(sigma_matrix.shape[0])
    ])
    return pareto_relation

# Відношення Парето P0 (асиметрична частина)
def get_P0_pareto_relation(sigma_matrix, n_alternatives):
    # Ініціалізація p0_pareto_relation заповненою нулями
    p0_pareto_relation = np.zeros((n_alternatives, n_alternatives), dtype=int)

    # Заповнення p0_pareto_relation
    for i in range(n_alternatives):
        for j in range(n_alternatives):
            if np.all(sigma_matrix[i, j] >= 0) and np.any(sigma_matrix[i, j] == 1):
                p0_pareto_relation[i, j] = 1  # Всі >= 0 та є хоча б одна 1
            else:
                p0_pareto_relation[i, j] = 0  # В іншому випадку 0
    return p0_pareto_relation


# Мажоритарне відношення
def majoritarian_relation(sigma_matrix, n_alternatives):
    majority_matrix = np.zeros((n_alternatives, n_alternatives), dtype=int)

    for i, x in enumerate(sigma_matrix):
        for j, y in enumerate(sigma_matrix):
            majority_matrix[i, j] = np.sum(sigma_matrix[i, j]) # рахуємо суму різниць оцінок

    # домінатними в мажоритарному відношенні переваги вважються ті альтернативи в яких сума
    # оцінок в sigma_matrix строго більше за 0
    majority_matrix = np.where(majority_matrix > 0, 1, 0)

    return majority_matrix


def lexicographic_relation(evaluation_matrix, n_alternatives, n_criterias, strict_order):
    # Перетасовуємо критерії між собою за V1
    evaluation_matrix_V1 = evaluation_matrix[:, strict_order]
    print(f"\nМатриця оцінок за критеріями в порядку важливості {strict_order}:")
    print(evaluation_matrix_V1)

    diff_matrix_V1 = get_criteria_diffs(evaluation_matrix_V1)
    sigma_matrix_V1 = get_sigma_matrix(diff_matrix_V1)

    lexicographic_relation = np.zeros((n_alternatives, n_alternatives), dtype=int)

    for i in range(n_alternatives):
        for j in range(n_alternatives):
            sigma_array = sigma_matrix_V1[i, j]
            for m in range(n_criterias):
                if (m == 0 and sigma_array[m] == 1) or (np.all(sigma_array[:m] == 0) and sigma_array[m] == 1):
                    lexicographic_relation[i, j] = 1
    
    return lexicographic_relation


def berezovsky_relation(evaluation_matrix, n_alternatives, quasi_order):
    print(f"\nВідношення квазіпорядку на мн-ні критеріїв: {quasi_order}")

    for l_class, l_criterias in enumerate(quasi_order):
        evaluation_matrix_V2 = evaluation_matrix[:, l_criterias]
        diff_matrix_V2 = get_criteria_diffs(evaluation_matrix_V2)
        sigma_matrix_V2 = get_sigma_matrix(diff_matrix_V2)
        p0_matrix = get_P0_pareto_relation(sigma_matrix_V2, n_alternatives)
        i0_matrix = get_I0_pareto_relation(sigma_matrix_V2, n_alternatives)
        n0_matrix = get_N0_pareto_relation(sigma_matrix_V2, n_alternatives)

        if l_class == 0:
            pB = p0_matrix
            iB = i0_matrix
            nB = n0_matrix
        else:
            for i in range(n_alternatives):
                for j in range(n_alternatives):
                    pB[i][j] = (p0_matrix[i][j] == 1 and (pB[i][j] == 1 or nB[i][j] == 1 or iB[i][j] == 1)) or (iB[i][j] == 1 and pB[i][j] == 1)

    return pB


def podinovsky_relation(evaluation_matrix, n_alternatives):
    evaluation_matrix_sorted = evaluation_matrix
    for i in range(n_alternatives):
        evaluation_matrix_sorted[i] = np.array(sorted(evaluation_matrix[i], reverse=True))
    print()
    print("\nОцінки відсортовані в порядку спадання:")
    print(evaluation_matrix_sorted)
    diff_matrix_sorted = get_criteria_diffs(evaluation_matrix_sorted)
    sigma_matrix_sorted = get_sigma_matrix(diff_matrix_sorted)
    rP_matrix = get_R0_pareto_relation(sigma_matrix_sorted, n_alternatives)

    return rP_matrix


# Головна функція
def main():
    methods = {
        "Pareto": lambda **kwargs: get_R0_pareto_relation(sigma_matrix, n_alternatives),
        "Majoritarian": lambda **kwargs: majoritarian_relation(sigma_matrix, n_alternatives),
        "Lexicographic": lambda **kwargs: lexicographic_relation(evaluation_matrix, n_alternatives, n_criterias, strict_order),
        "Berezovsky": lambda **kwargs: berezovsky_relation(evaluation_matrix, n_alternatives, quasi_order),
        "Podynovsky": lambda **kwargs: podinovsky_relation(evaluation_matrix, n_alternatives)
    }

    if len(methods) == 1:
        input_file = f"{str(list(methods.keys())[0]).lower()}_test.txt"
    elif len(methods) == 5:
        input_file = "Варіант №60.txt"
    input_file = "Варіант №60.txt"

    evaluation_matrix, strict_order, quasi_order = read_input(f"multi_criteria_decision_making/input/{input_file}")
    print("\nМатриця оцінок за критеріями множини K:")
    print(evaluation_matrix)
    print()

    diff_matrix = get_criteria_diffs(evaluation_matrix) # матриця різниць оцінок за критеріями
    sigma_matrix = get_sigma_matrix(diff_matrix) # сігма матриця

    n_alternatives = sigma_matrix.shape[0]  # Кількість альтернатив
    n_criterias = evaluation_matrix.shape[1] # Кількість критерій
    
    relations = {}
    for method, func in methods.items():
        relation_matrix = func(sigma_matrix=sigma_matrix, n_alternatives=n_alternatives,
                                n_criterias=n_criterias, evaluation_matrix=evaluation_matrix,
                                strict_order=strict_order, quasi_order=quasi_order)
        relations[method] = relation_matrix
    
    # Знаходимо оптимальні розвʼязки за допомогою метода Неймана-Моргенштерна або K-оптимізації
    optimal_sets = find_optimal_solution(alien_relations=relations)

    # Записуємо відношення до файлу
    output_file_path = "multi_criteria_decision_making/output/result.txt"
    with open(output_file_path, "w") as file:
        relation_index = 1  # Start numbering relations
        for key, relation in relations.items():
            # Write relation index
            file.write(f"{relation_index}\n")
            
            # Write the relation matrix
            for row in relation:
                row_str = " ".join(map(str, row))
                file.write(f"{row_str}\n")
            
            # Increment the relation index
            relation_index += 1


if __name__ == "__main__":
    main()