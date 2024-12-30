import numpy as np


# Клас для зручного зберігання та користування вхідними даними
class InputData:
    def __init__(self,
                alts_evals: np.ndarray,
                weights: np.ndarray):
        self.alts_evals = alts_evals # матриця оцінок альтернатив за критеріями
        self.weights = np.array(weights) # вагові коефіцієнти


# Зчитуємо дані з файлу
def read_input(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Зчитуємо оцінки альтернатив за критеріями
    evaluation_matrix = []
    line_index = 0
    for line in lines:
        if all(char.isdigit() or char.isspace() for char in line.strip()):
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
            weight_coefficients = list(map(float, lines[i + 1].strip().split()))
            line_index = i + 2
            break

    return InputData(evaluation_matrix, weight_coefficients)


def display_matrix(matrix: np.ndarray):
    for row in matrix:
        display_array(row)

def display_array(array: np.ndarray):
    print(" ".join(map(lambda x: f'{x:7.4f}', array)))


def calculate_similarity(alts_evals: np.ndarray, weights: np.ndarray):
    normalized_scores = normalize_scores(alts_evals)
    print("\nНормалізовані оцінки:")
    display_matrix(normalized_scores)

    weighted_scores = apply_weights(normalized_scores, weights)
    print("\nЗважені оцінки:")
    display_matrix(weighted_scores)

    ideal_point = np.max(weighted_scores, axis=0)
    anti_ideal_point = np.min(weighted_scores, axis=0)

    print("\nУтопічна точка (PIS):")
    display_array(ideal_point)
    print("\nАнтиутопічна точка (NIS):")
    display_array(anti_ideal_point)

    distances_to_pis = compute_distances(weighted_scores, ideal_point)
    distances_to_nis = compute_distances(weighted_scores, anti_ideal_point)

    print()
    print(distances_to_pis)
    print()
    print(distances_to_nis)
    print()

    print("\nВідстані альтернатив до PIS (D*) та до NIS (D-)")
    print(f"{'Альт':<4} {'D*':<7} {'D-':<7}")
    for i, (d_pis, d_nis) in enumerate(zip(distances_to_pis, distances_to_nis)):
        print(f"A{i + 1:<2}: {d_pis:<7.4f} {d_nis:<7.4f}")

    similarity_scores = compute_similarity(distances_to_pis, distances_to_nis)
    return similarity_scores

def normalize_scores(alts_evals: np.ndarray):
    column_sums = np.sqrt(np.sum(alts_evals**2, axis=0))
    return alts_evals / column_sums

def apply_weights(normalized_scores: np.ndarray, weights: np.ndarray):
    return weights * normalized_scores

def compute_distances(weighted_scores: np.ndarray, reference_point: np.ndarray):
    distances = np.sqrt(np.sum((weighted_scores - reference_point) ** 2, axis=1))
    return distances

def compute_similarity(distances_to_pis: np.ndarray, distances_to_nis: np.ndarray):
    return distances_to_nis / (distances_to_pis + distances_to_nis)

def rank_alternatives(similarity_scores: np.ndarray):
    ranked_indices = np.argsort(similarity_scores)[::-1]
    ranked_scores = [(i + 1, similarity_scores[i]) for i in ranked_indices]
    return ranked_scores


def main():
    file_name = "Варіант №60 умова.txt"
    input_data = read_input(f"multi_criteria_dm_topsis/input/{file_name}")

    print("\nТаблиця оцінок альтернатив за критеріями:")
    display_matrix(input_data.alts_evals)

    print("Вагові коефіцієнти критеріїв:")
    display_array(input_data.weights)

    similarity_scores = calculate_similarity(input_data.alts_evals, input_data.weights)

    print("\nСтупінь наближеності до утопічної точки:")
    for i, score in enumerate(similarity_scores):
        number_with_star = f"C{i + 1}*"
        print(f"{number_with_star:<4}: {score:<7.4f}")

    ranked_alternatives = rank_alternatives(similarity_scores)
    print("\nРанжування на множині альтернатив:")
    print(" > ".join(f"A{alt[0]}" for alt in ranked_alternatives))

    best_choice = ranked_alternatives[0]
    print(f"\nНайкраща альтернатива є A{best_choice[0]} з оцінкою {best_choice[1]:.4f}.")


if __name__ == "__main__":
    main()