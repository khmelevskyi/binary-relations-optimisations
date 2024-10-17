import numpy as np

from methods.read_binary_relations import read_binary_relations_from_txt
from methods.acyclicity_check import adj_matrix_to_adj_list, is_acyclic_dfs
from methods.neumann_morgenstern import Neumann_Morgenstern_optimization


file_path = 'data/Варіант №60.txt'
relations = read_binary_relations_from_txt(file_path)
print("Number of R: {}".format(len(relations)))

for relation_name, R_matrix in relations.items():
    adj_list = adj_matrix_to_adj_list(adj_matrix=R_matrix) # get outgoing sets
    
    if is_acyclic_dfs(adj_list):
        solution = Neumann_Morgenstern_optimization(R_matrix)
        print("____{}____".format(relation_name))
        print("Xнм: {}".format(solution))
        # visualize to prove that множина C0 є внутрішньо і зовнішньо стійкою
    else:
        pass
        # parameters = k_optimization(R_array)
        # print_information_k_opt(parameters, relation_name)