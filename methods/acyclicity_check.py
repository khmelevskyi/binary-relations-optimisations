import numpy as np
from numpy._typing import NDArray
from collections import deque, defaultdict


# Function to convert adjacency matrix to adjacency list
# get outgoing set
def adj_matrix_to_adj_list(adj_matrix: NDArray) -> 'dict[int, list[int]]':
    graph = {}
    n = adj_matrix.shape[0]
    for i in range(n):
        graph[i] = (np.where(adj_matrix[i] == 1)[0]).tolist()
    return graph


def is_acyclic_dfs(graph: 'dict[int, list[int]]'):
    """
    Check if the directed graph is acyclic using DFS.
    Time Complexity: O(V + E)

    Parameters:
    - graph: Dict[int, List[int]] representing adjacency list.

    Returns:
    - True if acyclic, False otherwise.
    """
    visited = set()
    rec_stack = set()

    def dfs(v):
        visited.add(v)
        rec_stack.add(v)
        for neighbour in graph.get(v, []):
            if neighbour not in visited:
                if not dfs(neighbour):
                    return False
            elif neighbour in rec_stack:
                return False
        rec_stack.remove(v)
        return True

    for node in graph:
        if node not in visited:
            if not dfs(node):
                return False
    return True


def is_acyclic_kahn(graph: 'dict[int, list[int]]', num_vertices: int):
    """
    Check if the directed graph is acyclic using Kahn's algorithm.
    Time Complexity: O(V + E)

    Parameters:
    - graph: Dict[int, List[int]] representing adjacency list.
    - num_vertices: Total number of vertices in the graph.

    Returns:
    - True if acyclic, False otherwise.
    """
    in_degree = defaultdict(int)
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    # Initialize queue with nodes having in-degree 0
    queue = deque([u for u in range(num_vertices) if in_degree[u] == 0])

    count = 0
    while queue:
        u = queue.popleft()
        count += 1
        for v in graph.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return count == num_vertices



if __name__ == "__main__":
    import time
    from read_binary_relations import read_binary_relations_from_txt

    file_path = 'data/Варіант №60.txt'  # Replace with your actual file path
    relations = read_binary_relations_from_txt(file_path)

    # Benchmarking Function
    def benchmark_algorithms(relations: 'dict[str, NDArray]') -> 'dict[str, dict]':
        results = {}
        for relation_name, matrix in relations.items():
            # print(f"\nBenchmarking {relation_name}:")

            # Convert to adjacency list
            graph = adj_matrix_to_adj_list(matrix)
            num_vertices = matrix.shape[0]

            # Measure DFS-Based
            start_time = time.time()
            acyclic_dfs = is_acyclic_dfs(graph)
            time_dfs = time.time() - start_time
            # print(f"  DFS-Based: {'Acyclic' if acyclic_dfs else 'Cyclic'}, Time: {time_dfs:.6f} seconds")

            # Measure Kahn's Algorithm
            start_time = time.time()
            acyclic_kahn = is_acyclic_kahn(graph, num_vertices)
            time_kahn = time.time() - start_time
            # print(f"  Kahn's Algorithm: {'Acyclic' if acyclic_kahn else 'Cyclic'}, Time: {time_kahn:.6f} seconds")

            # Store results
            results[relation_name] = {
                'DFS': (acyclic_dfs, time_dfs),
                'Kahn': (acyclic_kahn, time_kahn)
            }
        return results

    benchmark_results = benchmark_algorithms(relations)

    # Optionally, analyze results
    print("\nBenchmark Results Summary:")
    for relation, algos in benchmark_results.items():
        print(f"\n{relation}:")
        for algo, (acyclic, exec_time) in algos.items():
            status = 'Acyclic' if acyclic else 'Cyclic'
            print(f"  {algo}:\t {status}, Time: {exec_time:.6f} seconds")
