from methods.k_optimisation import k_optimization
from methods.neumann_morgenstern import Neumann_Morgenstern_optimization
from methods.read_binary_relations import read_binary_relations_from_txt
from methods.acyclicity_check import adj_matrix_to_adj_list, is_acyclic_dfs


file_path = 'data/Варіант №60.txt'
relations = read_binary_relations_from_txt(file_path)
print("К-сть Бінарних Відношень: {}".format(len(relations)))

for relation_name, R_matrix in relations.items():
    adj_list = adj_matrix_to_adj_list(adj_matrix=R_matrix) # get outgoing sets
    
    if is_acyclic_dfs(adj_list):
        print(f"БВ {relation_name} є ациклічним")
        solution = Neumann_Morgenstern_optimization(R_matrix)
        print("____{}____".format(relation_name))
        print("Xнм: {}".format(solution))
        # visualize to prove that множина C0 є внутрішньо і зовнішньо стійкою
    else:
        print(f"БВ {relation_name} є не ациклічним")
        parameters = k_optimization(R_matrix)
        print("____{}____".format(relation_name))
        print(f"1-max: {parameters['1_max']}   1-opt: {parameters['1_opt']}")
        print(f"2-max: {parameters['2_max']}   2-opt: {parameters['2_opt']}")
        print(f"3-max: {parameters['3_max']}   3-opt: {parameters['3_opt']}")
        print(f"4-max: {parameters['4_max']}   4-opt: {parameters['4_opt']}")