import numpy as np


# Класи для зручного зберігання та користування даними
class InputData:
    def __init__(self,
                alts_evals: np.ndarray,
                weights: np.ndarray):
        self.alts_evals = alts_evals  # матриця оцінок альтернатив за критеріями
        self.weights = np.array(weights)  # вагові коефіцієнти

class VikorRankings:
    def __init__(self,
                 s: np.ndarray,
                 r: np.ndarray,
                 q: np.ndarray):
        self.s = s
        self.r = r
        self.q = q


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


# -- display functions --
def display_matrix(matrix: np.ndarray):
    for row in matrix:
        display_array(row)

def display_array(array: np.ndarray):
    print(" ".join(map(lambda x: f'{x:7.4f}', array)))

def print_q_ranking(q_ranking: np.ndarray):
    print("Q: ", end="")
    print(" > ".join(map(lambda e: f"A{e[0] + 1}", np.flip(q_ranking))))

def print_rankings(rankings: VikorRankings):
    print(f"\n{f'Альт': <4} {'Sj': <6} {'Rj': <6} {'Qj': <6}")
    for i in range(len(rankings.s)):
        print(f"{f'A{i + 1}:': <4} {rankings.s[i]:<6.4f} {rankings.r[i]:<6.4f} {rankings.q[i]:<6.4f}")

def print_sorted_rankings(sorted_rankings: VikorRankings):
    print(f"\n{'Sj': <14} {'Rj': <14} {'Qj': <14}")
    for i in range(len(sorted_rankings.s)):
        print(f"{f'A{sorted_rankings.s[i][0] + 1}': <3} ({sorted_rankings.s[i][1]:<6.4f})   "
              f"{f'A{sorted_rankings.r[i][0] + 1}': <3} ({sorted_rankings.r[i][1]:<6.4f})   "
              f"{f'A{sorted_rankings.q[i][0] + 1}': <3} ({sorted_rankings.q[i][1]:<6.4f})")
# -- /display functions --


# -- VIKOR internal functions --
# Нормалізація оцінок
def get_normalized_ratings(alternatives_evaluations: np.ndarray) -> np.ndarray:
    max_evaluations = np.max(alternatives_evaluations, axis=0)
    min_evaluations = np.min(alternatives_evaluations, axis=0)
    normalized_ratings = np.zeros(alternatives_evaluations.shape)
    for i in range(len(normalized_ratings)):
        for j in range(len(normalized_ratings[i])):
            normalized_ratings[i, j] = (np.abs(max_evaluations[j] - alternatives_evaluations[i, j])
                                        / np.abs(max_evaluations[j] - min_evaluations[j]))
    return normalized_ratings

# Розрахунок зважених оцінок
def get_weighted_ratings(normalized_ratings: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return normalized_ratings * weights

# Розрахунок Sj
def get_s(weighted_ratings: np.ndarray) -> np.ndarray:
    return np.sum(weighted_ratings, axis=1)

# Розрахунок Rj
def get_r(weighted_ratings: np.ndarray) -> np.ndarray:
    return np.max(weighted_ratings, axis=1)

# Розрахунок Qj
def get_q(s: np.ndarray, r: np.ndarray, v: float) -> np.ndarray:
    s_best = np.min(s)
    s_worst = np.max(s)
    r_best = np.min(r)
    r_worst = np.max(r)
    q = v * (s - s_best) / (s_worst - s_best) + (1 - v) * (r - r_best) / (r_worst - r_best)
    return q

# Сортування результатів
def sort_rankings(rankings: VikorRankings) -> VikorRankings:
    rankings_as_list = [rankings.s, rankings.r, rankings.q]
    for i in range(len(rankings_as_list)):
        rankings_as_list[i] = np.array(
            [(j, value) for j, value in enumerate(rankings_as_list[i])],
            dtype=[('index', int), ('value', float)])
        rankings_as_list[i] = np.flip(np.sort(rankings_as_list[i], order='value'))

    sorted_rankings = VikorRankings(rankings_as_list[0], rankings_as_list[1], rankings_as_list[2])
    return sorted_rankings

# Отримання компромісного рішення
def get_compromise_solution(sorted_rankings: VikorRankings) -> list:
    alternatives_count = len(sorted_rankings.q)
    best_alternative = sorted_rankings.q[alternatives_count - 1]
    pre_best_alternative = sorted_rankings.q[alternatives_count - 2]

    dq = 1 / (alternatives_count - 1)
    c1_is_satisfied = pre_best_alternative[1] - best_alternative[1] >= dq
    c2_is_satisfied = (sorted_rankings.s[alternatives_count - 1][0] == best_alternative[0]
                       or sorted_rankings.r[alternatives_count - 1][0] == best_alternative[0])

    compromise_solution = [best_alternative[0]]
    if not c1_is_satisfied:
        current_q_index = alternatives_count - 2
        while current_q_index >= 0 and sorted_rankings.q[current_q_index][1] - best_alternative[1] < dq:
            compromise_solution.append(sorted_rankings.q[current_q_index][0])
            current_q_index -= 1
    elif not c2_is_satisfied:
        compromise_solution.append(pre_best_alternative[0])

    print(f"C1 {'не ' if not c1_is_satisfied else ''}задовільняє")
    print(f"C2 {'не ' if not c2_is_satisfied else ''}задовільняє")

    return compromise_solution
# -- /VIKOR internal functions --


# Результати VIKOR з логуванням
def get_vikor_results_with_logs(data: InputData, v: float = 0.5):
    normalized_ratings = get_normalized_ratings(data.alts_evals)
    print("\nНормалізовані оцінки:")
    display_matrix(normalized_ratings)

    weighted_ratings = get_weighted_ratings(normalized_ratings, data.weights)
    print("\nЗважені оцінки:")
    display_matrix(weighted_ratings)

    s = get_s(weighted_ratings)
    r = get_r(weighted_ratings)
    q = get_q(s, r, v)

    rankings = VikorRankings(s, r, q)
    print_rankings(rankings)

    sorted_rankings = sort_rankings(rankings)
    print_sorted_rankings(sorted_rankings)
    print_q_ranking(sorted_rankings.q)

    compromise_solution = get_compromise_solution(sorted_rankings)
    print(f"Компромісне рішення: {', '.join(map(lambda e: f'A{e + 1}', compromise_solution))}")


# Експерименти з впливом V
def execute_v_influence_experiments(data: InputData):
    print("\n\nЕксперименти з впливом v:")
    for v in np.arange(0, 1 + 0.1, 0.1):
        print("\n\n----------")
        print(f"v = {v:.1f}")
        normalized_ratings = get_normalized_ratings(data.alts_evals)
        weighted_ratings = get_weighted_ratings(normalized_ratings, data.weights)
        s = get_s(weighted_ratings)
        r = get_r(weighted_ratings)
        q = get_q(s, r, v)

        rankings = VikorRankings(s, r, q)
        sorted_rankings = sort_rankings(rankings)
        print_sorted_rankings(sorted_rankings)
        print_q_ranking(sorted_rankings.q)

        print("S: ", end="")
        print(", ".join(map(lambda e: f"A{e[0] + 1}", sorted_rankings.s)))
        print("R: ", end="")
        print(", ".join(map(lambda e: f"A{e[0] + 1}", sorted_rankings.r)))
        print("Q: ", end="")
        print(", ".join(map(lambda e: f"A{e[0] + 1}", sorted_rankings.q)))
        
        compromise_solution = get_compromise_solution(sorted_rankings)
        print(f"Компромісне рішення: {', '.join(map(lambda e: f'A{e + 1}', compromise_solution))}")


# Головна функція
def main():
    file_name = "Варіант №60 умова.txt"
    input_data = read_input(f"multi_criteria_dm_vikor/input/{file_name}")

    print("\nТаблиця оцінок альтернатив за критеріями:")
    display_matrix(input_data.alts_evals)
    print("Вагові коефіцієнти критеріїв:")
    display_array(input_data.weights)
    print()

    get_vikor_results_with_logs(input_data)
    execute_v_influence_experiments(input_data)


if __name__ == "__main__":
    main()
