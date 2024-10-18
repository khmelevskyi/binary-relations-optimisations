from methods.k_optimisation import k_optimization
from methods.graph_visualization import display_relation_graph
from methods.neumann_morgenstern import Neumann_Morgenstern_optimization
from methods.read_binary_relations import read_binary_relations_from_txt
from methods.acyclicity_check import adj_matrix_to_adj_list, is_acyclic_dfs
from methods.matrix_separate_symmetric_asymmetric import display_str_matrix


file_path = 'data/Варіант №60.txt'
relations = read_binary_relations_from_txt(file_path)
print("К-сть Бінарних Відношень: {}".format(len(relations)))

for relation_name, R_matrix in relations.items():
    adj_list = adj_matrix_to_adj_list(adj_matrix=R_matrix) # get outgoing sets

    print()
    display_str_matrix(R_matrix)
    
    if is_acyclic_dfs(adj_list):
        print(f"БВ {relation_name} є ациклічним\n")
        solution = Neumann_Morgenstern_optimization(R_matrix)
        print(f"\n____{relation_name}____")
        print(f"Xнм: {solution}\n")
        # visualize to prove that множина C0 є внутрішньо і зовнішньо стійкою
        display_relation_graph(R_matrix, solution)
    else:
        print(f"БВ {relation_name} є не ациклічним\n")
        parameters = k_optimization(R_matrix)
        print(f"____{relation_name}____")
        print(f"1-max: {parameters['1_max']}   1-opt: {parameters['1_opt']}")
        print(f"2-max: {parameters['2_max']}   2-opt: {parameters['2_opt']}")
        print(f"3-max: {parameters['3_max']}   3-opt: {parameters['3_opt']}")
        print(f"4-max: {parameters['4_max']}   4-opt: {parameters['4_opt']}")