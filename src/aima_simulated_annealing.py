import random, math
from aima_nn_algs import path_cost
from aima_hill_climbing import random_swap
import time



def simulated_annealing(weights, alpha, initial_temp, max_iterations):
    n = weights.shape[0]
    
    current = list(range(n))
    random.shuffle(current)
    current = current + [current[0]]
    
    curr_cost = path_cost(weights, current)
    best_path, best_cost = current[:], curr_cost
    t = initial_temp
    history = [best_cost]
    
    for _ in range(max_iterations):
        neighbor_path = random_swap(current)
        neighbor_cost = path_cost(weights, neighbor_path)

        delta = neighbor_cost - curr_cost

        if delta <= 0 or random.random() < math.exp(-delta / max(t, 1e-12)):
            current, curr_cost = neighbor_path, neighbor_cost
            if curr_cost < best_cost:
                best_path, best_cost = current[:], curr_cost

        t *= alpha
        history.append(best_cost)

        if t < 1e-12:
            break

    return best_path, best_cost, history
        
if __name__ == "__main__":
    import sys, numpy as np
    
    if len(sys.argv) < 2:
        print("Usage: python aima_simulated_annealing.py <matrix_file> [alpha] [t0] [iters]")
        sys.exit(1)

    mat = sys.argv[1]
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.995
    t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    iters = int(sys.argv[4])   if len(sys.argv) > 4 else 20000

    W = np.loadtxt(mat)

    start_wall = time.time_ns()
    start_cpu = time.process_time_ns()
    path, cost, history = simulated_annealing(W, alpha=alpha, initial_temp=t0, max_iterations=iters)
    end_wall = time.time_ns()
    end_cpu = time.process_time_ns()

    print("===== Simulated Annealing (random swaps) =====")
    print(f"Matrix: {mat}")
    print(f"alpha={alpha}, t0={t0}, iters={iters}")
    print("Path:", path)
    print("Cost:", cost)
    print("History length:", len(history))
    print("Wall time (ns):", end_wall - start_wall)
    print("CPU time (ns):", end_cpu - start_cpu)