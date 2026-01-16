import random
from aima_nn_algs import path_cost
import time

def crossover(p1, p2):
    """Order Crossover (OX) on the permutation part (exclude last element)."""
    n = len(p1) - 1
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n

    # copy slice from p1 (inclusive a..b)
    child[a:b+1] = p1[a:b+1]

    # fill remaining positions in order of appearance from p2
    used = set(child[a:b+1])
    pos = (b + 1) % n
    for city in p2[:-1]:  # skip the duplicate last element
        if city in used:
            continue
        while child[pos] is not None:
            pos = (pos + 1) % n
        child[pos] = city
        pos = (pos + 1) % n

    child.append(child[0])  # close tour
    return child

def mutate(path, mutation_chance):
    """Swap-mutation on two indices in the permutation (exclude last)."""
    if random.random() < mutation_chance:
        n = len(path) - 1
        i, j = sorted(random.sample(range(n), 2))
        path[i], path[j] = path[j], path[i]
        path[-1] = path[0]  # re-close
    return path

def genetic_algorithm(weights, mutation_chance, pop_size, num_generations):
    n = weights.shape[0]

    population = []
    for _ in range(pop_size):
        tour = list(range(n))
        random.shuffle(tour)
        tour.append(tour[0])
        population.append((tour, path_cost(weights, tour)))

    best_path, best_cost = population[0][0][:], population[0][1]
    history = []  # best-so-far cost per generation

    for _ in range(num_generations):
        population.sort(key=lambda x: x[1])
        survivors = population[: max(2, pop_size // 2)]  # ensure ≥2 parents

        if survivors[0][1] < best_cost:
            best_cost = survivors[0][1]
            best_path = survivors[0][0][:]

        history.append(best_cost)

        new_population = survivors[:]
        while len(new_population) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            child = crossover(p1[0], p2[0])
            child = mutate(child, mutation_chance)
            new_population.append((child, path_cost(weights, child)))

        population = new_population

    return best_path, best_cost, history


if __name__ == "__main__":
    import sys, numpy as np
    if len(sys.argv) < 2:
        print("Usage: python aima_ga.py <matrix_file> [mutation] [pop_size] [generations]")
        sys.exit(1)

    mat = sys.argv[1]
    mutation = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    pop = int(sys.argv[3]) if len(sys.argv) > 3 else 80
    gens = int(sys.argv[4]) if len(sys.argv) > 4 else 200

    W = np.loadtxt(mat)

    start_wall = time.time_ns()
    start_cpu = time.process_time_ns()
    path, cost, history = genetic_algorithm(W, mutation, pop, gens)
    end_wall = time.time_ns()
    end_cpu = time.process_time_ns()

    print("===== Genetic Algorithm =====")
    print(f"Matrix: {mat}")
    print(f"mutation={mutation}, pop_size={pop}, generations={gens}")
    print("Path:", path)
    print("Cost:", cost)
    print("History length:", len(history))
    print("Wall time (ns):", end_wall - start_wall)
    print("CPU time (ns):", end_cpu - start_cpu)
