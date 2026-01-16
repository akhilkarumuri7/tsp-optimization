import sys
import time
import random
import numpy as np
from aima_nn_algs import path_cost

def random_swap(path):
    new_path = path[:]
    i, j = random.sample(range(1, len(path) - 1), 2)
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

def hill_climb(weights, num_restarts, max_iterations):
    n = weights.shape[0]
    best_path, best_cost = [], float("inf")
    history = []
    
    for _ in range(num_restarts):
        # random initial tour
        current = list(range(n))
        random.shuffle(current)
        current.append(current[0])
        current_cost = path_cost(weights, current)

        for _ in range(max_iterations):
            neighbor = random_swap(current)
            neighbor_cost = path_cost(weights, neighbor)

            # greedy accept: only keep improving neighbors
            if neighbor_cost < current_cost:
                current, current_cost = neighbor, neighbor_cost

            if current_cost < best_cost:
                best_cost = current_cost
                best_path = current

            history.append(best_cost)

    return best_path, best_cost, history
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hill_climb.py <matrix_file.txt> [num_restarts] [max_iterations]")
        sys.exit(1)

    mat_file = sys.argv[1]
    num_restarts = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    max_iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    W = np.loadtxt(mat_file)

    start_wall = time.time_ns()
    start_cpu = time.process_time_ns()
    path, cost, history = hill_climb(W, num_restarts=num_restarts, max_iterations=max_iterations)
    end_wall = time.time_ns()
    end_cpu = time.process_time_ns()


    print("===== Hill Climbing (random swaps) =====")
    print(f"Matrix: {mat_file}")
    print(f"num_restarts={num_restarts}, max_iterations={max_iterations}")
    print("Cost: ", cost)
    print("Path: ", path)
    print("History length: ", len(history))
    print("  Wall time (ns):", end_wall - start_wall)
    print("  CPU time (ns):", end_cpu - start_cpu)

