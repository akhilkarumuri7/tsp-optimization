import sys
import numpy as np
from src.aima_my_tsp import MyTSP
from aima.search import astar_search

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_astar.py <matrix_file>")
        sys.exit(1)

    matrix_file = sys.argv[1]
    mat = np.loadtxt(matrix_file)

    problem = MyTSP(mat)
    node = astar_search(problem, display=True)

    if node is None:
        print("No solution found.")
        return

    path = tuple(node.state)
    cost = problem.value(path)
    print("\n===== A* (TSP with MST heuristic) =====")
    print("Path:", path)
    print("Cost:", cost)

if __name__ == "__main__":
    main()
