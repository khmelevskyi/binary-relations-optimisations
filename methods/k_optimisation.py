import numpy as np
from numpy._typing import NDArray
from methods.matrix_separate_symmetric_asymmetric import separate_symmetric_asymmetric, display_str_matrix


def check_x(S: NDArray):
    max_k = np.array([])
    opt_k = np.array([])
    for i, x in enumerate(S, start=1):
        if np.any(S-x > 0): #формуємо множину k-макс. елементів
            continue
        max_k = np.append(max_k, i)
        if np.sum(x) == S.shape[0]: #формуємо множину k-опт. елементів
            opt_k = np.append(opt_k, i)
        
    return max_k, opt_k

def k_optimization(Rn: NDArray):
    I = (Rn == Rn.T)*Rn # формуємо симетричну частину
    P = Rn-I # формуємо асиметричну частину
    N = (Rn == Rn.T)-I # формуємо відношення непорівнюваності

    Rn_str = separate_symmetric_asymmetric(Rn)
    
    S1=I+P+N
    display_str_matrix(Rn_str, "S1")
    max_1, opt_1 = check_x(S1)

    S2=P+N
    S2_str = np.where(Rn_str == 'I', '0', Rn_str)
    display_str_matrix(S2_str, "S2")
    max_2, opt_2 = check_x(S2)
    
    S3=P+I
    S3_str = np.where(Rn_str == 'N', '0', Rn_str)
    display_str_matrix(S3_str, "S3")
    max_3, opt_3 = check_x(S3)
    
    S4=P
    S4_str = np.where(S3_str == 'I', '0', S3_str)
    display_str_matrix(S4_str, "S4")
    max_4, opt_4 = check_x(S4)
            
    parameters = {"1_max": max_1,
                 "1_opt": opt_1,
                 "2_max": max_2,
                 "2_opt": opt_2,
                 "3_max": max_3,
                 "3_opt": opt_3,
                 "4_max": max_4,
                 "4_opt": opt_4}
    
    return parameters
