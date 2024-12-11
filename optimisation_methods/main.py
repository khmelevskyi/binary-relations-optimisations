from optimisation_methods.k_optimisation import k_optimization
from optimisation_methods.graph_visualization import display_relation_graph
from optimisation_methods.neumann_morgenstern import Neumann_Morgenstern_optimization
from optimisation_methods.read_binary_relations import read_binary_relations_from_txt
from optimisation_methods.acyclicity_check import adj_matrix_to_adj_list, is_acyclic_dfs
from optimisation_methods.matrix_separate_symmetric_asymmetric import display_str_matrix
from optimisation_methods.matrix_separate_symmetric_asymmetric import classify_matrix, separate_symmetric_asymmetric


def run_search_for_optimal(relation_name, R_matrix):
    adj_list = adj_matrix_to_adj_list(adj_matrix=R_matrix) # get outgoing sets

    result = separate_symmetric_asymmetric(R_matrix)
    classify_matrix(result)
    
    if is_acyclic_dfs(adj_list):
        print(f"БВ {relation_name} є ациклічним. Використовуємо метод Неймана-Моргенштерна\n")
        solution = Neumann_Morgenstern_optimization(R_matrix)
        print(f"\n____{relation_name}____")
        print(f"Xнм: {solution}\n")
        # visualize to prove that множина C0 є внутрішньо і зовнішньо стійкою
        display_relation_graph(R_matrix, solution)
        return solution
    else:
        print(f"БВ {relation_name} є не ациклічним. Використовуємо метод K-оптимізації\n")
        parameters = k_optimization(R_matrix)
        print(f"____{relation_name}____")
        print(f"1-max: {parameters['1_max']}   1-opt: {parameters['1_opt']}")
        print(f"2-max: {parameters['2_max']}   2-opt: {parameters['2_opt']}")
        print(f"3-max: {parameters['3_max']}   3-opt: {parameters['3_opt']}")
        print(f"4-max: {parameters['4_max']}   4-opt: {parameters['4_opt']}")
        return parameters


def find_optimal_solution(file_path: str = None, alien_relations: dict = None):
    
    if file_path is not None:
        relations = read_binary_relations_from_txt(file_path)
        print("К-сть Бінарних Відношень: {}".format(len(relations)))
    elif alien_relations is not None:
        relations = alien_relations

    optimal_sets = []
    for relation_name, R_matrix in relations.items():
        print(f"\n____{relation_name}____")
        display_str_matrix(R_matrix)

        optimal_set = run_search_for_optimal(relation_name, R_matrix)
        optimal_sets.append(optimal_set)
        
    return optimal_sets

if __name__ == "__main__":
    file_path = 'data/Варіант №60.txt'
    find_optimal_solution(file_path=file_path)