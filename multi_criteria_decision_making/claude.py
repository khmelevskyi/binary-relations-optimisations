import numpy as np
from typing import List, Dict, Tuple


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


class MultiCriteriaRelations:
    def __init__(self, criteria_matrix: np.ndarray, 
                 V1_ranking: List[int] = None, 
                 V2_classes: List[List[int]] = None):
        """
        Ініціалізація класу для роботи з багатокритеріальним вибором
        
        :param criteria_matrix: Матриця оцінок критеріїв
        :param V1_ranking: Строге впорядкування критеріїв за важливістю
        :param V2_classes: Класи рівноважливих критеріїв
        """
        self.criteria_matrix = criteria_matrix
        self.V1_ranking = V1_ranking or list(range(criteria_matrix.shape[1]))
        self.V2_classes = V2_classes or [[i] for i in range(criteria_matrix.shape[1])]
        self.m = criteria_matrix.shape[1]  # Кількість критеріїв
        self.n = criteria_matrix.shape[0]  # Кількість альтернатив

    def pareto_relation(self) -> Dict[str, np.ndarray]:
        """
        Реалізація відношення Парето
        
        :return: Словник з матрицями відношень
        """
        pareto_matrix = np.zeros((self.n, self.n), dtype=int)
        
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                
                # Перевірка умов домінування за Парето
                better_count = 0
                equal_count = 0
                
                for k in range(self.m):
                    if self.criteria_matrix[i, k] > self.criteria_matrix[j, k]:
                        better_count += 1
                    elif self.criteria_matrix[i, k] == self.criteria_matrix[j, k]:
                        equal_count += 1
                
                if better_count > 0 and better_count + equal_count == self.m:
                    pareto_matrix[i, j] = 1
        
        return {
            'matrix': pareto_matrix,
            'pareto_set': np.where(np.sum(pareto_matrix, axis=1) == 0)[0]
        }

    def majoritarian_relation(self) -> Dict[str, np.ndarray]:
        """
        Реалізація мажоритарного відношення
        
        :return: Словник з матрицями відношень
        """
        majoritarian_matrix = np.zeros((self.n, self.n), dtype=int)
        
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                
                better_count = sum(
                    self.criteria_matrix[i, k] > self.criteria_matrix[j, k]
                    for k in range(self.m)
                )
                
                if better_count > self.m // 2:
                    majoritarian_matrix[i, j] = 1
        
        return {
            'matrix': majoritarian_matrix,
            'best_alternatives': np.argmax(np.sum(majoritarian_matrix, axis=1))
        }

    def lexicographic_relation(self) -> Dict[str, int]:
        """
        Реалізація лексикографічного відношення
        
        :return: Словник з кращою альтернативою
        """
        # Сортування критеріїв за важливістю (V1_ranking)
        sorted_criteria = sorted(
            enumerate(self.V1_ranking), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        best_alternative = 0
        for criterion_idx, _ in sorted_criteria:
            criterion_values = self.criteria_matrix[:, criterion_idx]
            best_alternatives = np.where(
                criterion_values == np.max(criterion_values)
            )[0]
            
            if len(best_alternatives) == 1:
                best_alternative = best_alternatives[0]
                break
            elif len(best_alternatives) > 1:
                self.criteria_matrix = self.criteria_matrix[best_alternatives]
                
        return {
            'best_alternative': best_alternative,
            'best_value': self.criteria_matrix[best_alternative]
        }

    def berezovsky_relation(self) -> Dict[str, np.ndarray]:
        """
        Реалізація відношення Березовського
        
        :return: Словник з матрицями відношень
        """
        def pareto_comparison(x, y, class_idx):
            """Порівняння за Парето в рамках класу"""
            class_criteria = [
                c for c in range(self.m) 
                if any(c in cls for cls in self.V2_classes[class_idx])
            ]
            
            better_count = 0
            equal_count = 0
            
            for k in class_criteria:
                if x[k] > y[k]:
                    better_count += 1
                elif x[k] == y[k]:
                    equal_count += 1
            
            return better_count > 0 and better_count + equal_count == len(class_criteria)
        
        berezovsky_matrix = np.zeros((self.n, self.n), dtype=int)
        
        for class_idx in range(len(self.V2_classes)):
            for i in range(self.n):
                for j in range(self.n):
                    if pareto_comparison(
                        self.criteria_matrix[i], 
                        self.criteria_matrix[j], 
                        class_idx
                    ):
                        berezovsky_matrix[i, j] = 1
        
        return {
            'matrix': berezovsky_matrix,
            'best_alternatives': np.where(np.sum(berezovsky_matrix, axis=1) == 0)[0]
        }

    def podinovsky_relation(self) -> Dict[str, int]:
        """
        Реалізація відношення Подиновського
        
        :return: Словник з кращою альтернативою
        """
        def symmetric_transform(x, i, j):
            """Симетрична перестановка координат"""
            x_copy = x.copy()
            x_copy[i], x_copy[j] = x_copy[j], x_copy[i]
            return x_copy
        
        def check_relation_chain(x, y):
            """Перевірка ланцюжка перетворень"""
            # TODO: Деталізувати логіку перевірки ланцюжка
            return False
        
        best_alternative = 0
        for i in range(self.n):
            for j in range(self.n):
                if i != j and check_relation_chain(
                    self.criteria_matrix[i], 
                    self.criteria_matrix[j]
                ):
                    best_alternative = i
                    break
        
        return {
            'best_alternative': best_alternative,
            'best_value': self.criteria_matrix[best_alternative]
        }

# Приклад використання
if __name__ == "__main__":
    # Приклад матриці критеріїв
    # criteria_matrix = np.array([
    #     [2, 1, 8, 9, 6, 5, 7, 3, 4, 10, 11, 12],
    #     [10, 4, 9, 9, 7, 8, 5, 6, 3, 2, 1, 11],
    #     [8, 1, 6, 7, 5, 9, 10, 4, 2, 3, 12, 11]
    # ])
    
    # # Приклад порівнюваності
    # V1_ranking = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # V2_classes = [[0, 1], [2, 3], [4, 5, 6], [7, 8, 9], [10, 11]]

    criteria_matrix, V1_ranking, V2_classes = read_input("input/pareto_test.txt")
    
    mcr = MultiCriteriaRelations(
        criteria_matrix, 
        V1_ranking, 
        V2_classes
    )
    
    print("Парето:\n", mcr.pareto_relation())
    # print("\n\nМажоритарне:\n", mcr.majoritarian_relation())
    # print("\n\nЛексикографічне:\n", mcr.lexicographic_relation())
    # print("\n\nБерезовського:\n", mcr.berezovsky_relation())
    # print("\n\nПодиновського:\n", mcr.podinovsky_relation())