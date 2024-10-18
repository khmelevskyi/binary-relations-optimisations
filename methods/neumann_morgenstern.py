import numpy as np
from numpy._typing import NDArray


S, S_delta, Q = dict(), dict(), dict()

# верхні перерізи альтернатив
def get_incoming_sets(R: NDArray) -> 'dict[int, list]':
    incoming_sets = {}
    R_t = R.T
    for i in range(R.shape[0]):
        incoming_sets[i] = (np.where(R_t[i] == 1)[0]).tolist()

    # Increase both keys and values by 1 (to match the usual binary relation representation)
    incoming_sets = {key + 1: [val + 1 for val in values] for key, values in incoming_sets.items()}
    return incoming_sets


def get_S0(incoming_sets: 'dict[int, list]'):
    s0 = [key for key, value in incoming_sets.items() if len(value) == 0]

    S[0] = s0
    S_delta[0] = s0


def get_all_S(R: NDArray, incoming_sets: 'dict[int, list]'):
    last_key = max(S.keys())
    last_array = S[last_key]

    incoming_sets_residual = incoming_sets.copy()
    # Remove the already added keys from temporary incoming_sets_residual dict
    for key in last_array:
        del incoming_sets_residual[key]

    while len(S[last_key]) != R.shape[0]:
        # Get the array of the last key in S (the most recently added one)
        last_key = max(S.keys())
        last_array = S[last_key]

        # To store the new keys to be added in this iteration
        new_keys = []

        # Check which keys from incoming_sets can be added to S and S_delta
        for key, incoming in incoming_sets_residual.items():
            if np.all(np.isin(incoming, last_array)):
                new_keys.append(key)

        # If no new keys can be added, break the loop
        if not new_keys:
            break

        # Add the new keys to S_delta and S
        new_key_S = max(S.keys()) + 1  # New key for S
        new_key_S_delta = max(S_delta.keys()) + 1  # New key for S_delta

        # Add the new keys to S_delta
        S_delta[new_key_S_delta] = new_keys

        # In S, append the new keys to the last key's elements
        new_elements_for_S = list(set(last_array + new_keys))
        S[new_key_S] = new_elements_for_S

        # Remove the already added keys from temporary incoming_sets_residual dict
        for key in new_keys:
            del incoming_sets_residual[key]


def get_all_Q(incoming_sets: 'dict[int, list]'):
    S_amount = len(S)

    for current_i in range(1, S_amount):
        # To store the new keys to be added in this iteration
        new_keys = []

        # Check which keys from S_delta can be added to Q
        for alt_node in S_delta[current_i]:
            if np.all(~np.isin(incoming_sets[alt_node], Q[current_i-1])):
                new_keys.append(alt_node)

        # Add the new keys to Q
        Q[current_i] = list(set(Q[current_i-1] + new_keys))


def Neumann_Morgenstern_optimization(R: NDArray) -> list:
    incoming_sets = get_incoming_sets(R) #знаходимо верхні перерізи для всіх альтернатив
    
    get_S0(incoming_sets) #S0

    get_all_S(R, incoming_sets) #STEP 1

    Q[0] = S[0] #Q0
    get_all_Q(incoming_sets) #STEP 2

    C0: list = Q[max(Q.keys())]
    return C0
