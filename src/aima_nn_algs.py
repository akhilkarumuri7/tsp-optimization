import sys
import numpy as np
import random
import time

def nearest_neighbor(weights):
    n = weights.shape[0]
    visited = {0}                   # keep track of visited cities
    path = [0]                      # start at city 0
    total_cost = 0
    
    while len(visited) < n:
        curr = path[-1]             # current city
        min_weight = float('inf')
        
        # find the nearest unvisited city
        next_city = None
        for city in range(weights.shape[0]):
            if city not in visited:
                if weights[curr, city] < min_weight:
                    min_weight = weights[curr, city]
                    next_city = city
                    
        path.append(next_city)      # add next city to path
        visited.add(next_city)      # mark it visited
        total_cost += min_weight    # add to total cost
        
    # return to start city
    total_cost += weights[path[-1], 0]
    path.append(0) 
    return (path, total_cost)
            
def path_cost(weights, path):
    cost = 0
    for i in range(len(path) - 1):
        cost += weights[path[i], path[i + 1]]
    return cost

def two_opt(weights, path):
    # takes in a nearest-neighbor path and tries to improve it via 2-opt
    n = len(path)
    improved = True
    cost = path_cost(weights, path)
    
    while improved:
        improved = False
        for i in range(1, n - 3):
            for j in range(i + 2, n):
                # calculate gain/loss from reversing segment [i:j]
                delta = ((weights[path[i-1], path[j-1]] + weights[path[i], path[j]])
                - (weights[path[i-1], path[i]] + weights[path[j-1], path[j]]))
                
                # if we found an improvement, swap and update cost
                if delta < 0: 
                    path[i:j] = reversed(path[i:j])
                    cost += delta
                    improved = True
                    break
            if improved:
                break
    return path, cost
                    
def rrnn(weights, k, num_repeats):
    n = weights.shape[0]
    best_path, best_cost = [], float("inf")

    for _ in range(num_repeats):
        visited = {0}
        path = [0]
        total_cost = 0

        while len(visited) < n:
            curr = path[-1]
            
            # collect unvisited cities and sort by distance from current city
            def distance_from_curr(city):
                return weights[curr, city]
        
            unvisited = []
            for city in range(n):
                if city not in visited:
                    unvisited.append(city)
                    
            unvisited.sort(key=distance_from_curr)
            
            num_cand = min(k, len(unvisited))
            candidates = unvisited[:num_cand]
            
            # randomly pick a city from the k nearest unvisited cities
            next_city = random.choice(candidates)
            total_cost += weights[curr, next_city]
            path.append(next_city)
            visited.add(next_city)
        
        # return to start city
        total_cost += weights[path[-1], 0]
        path.append(0)

        # apply 2-opt to the candidate path
        cand_path, cand_cost = two_opt(weights, path.copy())

        if cand_cost < best_cost:
            best_cost = cand_cost
            best_path = cand_path
            
    return best_path, best_cost


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aima_nn_algs.py <matrix_file>")
        sys.exit(1)

    matrix_file = sys.argv[1]
    weights = np.loadtxt(matrix_file)


    # Nearest Neighbor
    start_wall = time.time_ns()
    start_cpu = time.process_time_ns()
    nn_path, nn_cost = nearest_neighbor(weights)
    end_wall = time.time_ns()
    end_cpu = time.process_time_ns()
    print("===== Nearest Neighbor =====")
    print("  Path:", nn_path)
    print("  Cost:", nn_cost)
    print("  Wall time (ns):", end_wall - start_wall)
    print("  CPU time (ns):", end_cpu - start_cpu)


    # Nearest Neighbor + 2-Opt
    start_wall = time.time_ns()
    start_cpu = time.process_time_ns()
    opt_path, opt_cost = two_opt(weights, nn_path)
    end_wall = time.time_ns()
    end_cpu = time.process_time_ns()
    print("\n===== Nearest Neighbor + 2-Opt =====")
    print("  Path:", list(opt_path))
    print("  Cost:", opt_cost)
    print("  Wall time (ns):", end_wall - start_wall)
    print("  CPU time (ns):", end_cpu - start_cpu)
    
    # Repeated Random Nearest Neighbor
    start_wall = time.time_ns()
    start_cpu = time.process_time_ns()
    rrnn_path, rrnn_cost = rrnn(weights, 2, 20)
    end_wall = time.time_ns()
    end_cpu = time.process_time_ns()
    print("\n===== RRNN =====")
    print("  Path:", rrnn_path)
    print("  Cost:", rrnn_cost)
    print("  Wall time (ns):", end_wall - start_wall)
    print("  CPU time (ns):", end_cpu - start_cpu)
    