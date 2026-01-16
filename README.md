# CMSC 421 Project 1 — Traveling Salesman Problem (TSP)
**Informed Search and Local Optimization Algorithms**

This repository contains my implementation and experimental analysis for **CMSC 421 Project 1**, which explores multiple algorithms for solving the **Traveling Salesman Problem (TSP)**. The project focuses on comparing solution quality, runtime, CPU time, scalability, and the impact of hyperparameters across deterministic, randomized, and informed search methods.

---

## Algorithms Implemented

### Part I — Greedy & Randomized Algorithms
- **Nearest Neighbor (NN)**
- **Nearest Neighbor + 2-Opt (NN2O)**
- **Repeated Random Nearest Neighbor (RRNN)**  
  - Hyperparameters: `k` (candidate pool size), `num_repeats`

### Part II — Informed Search
- **A\*** with **Minimum Spanning Tree (MST) heuristic**
  - Guaranteed optimal solution
  - Used as a baseline for cost and runtime ratios

### Part III — Local Search & Evolutionary Algorithms
- **Hill Climbing** (random restarts)
- **Simulated Annealing**
- **Genetic Algorithm**

---

## Repository Structure

```
tsp_optimization/
├── data/                   # Input adjacency matrices
├── src/                    # Algorithm implementations
│   ├── aima_nn_algs.py
│   ├── aima_my_tsp.py      # A* with MST heuristic
│   ├── aima_hill_climbing.py
│   ├── aima_simulated_annealing.py
│   └── aima_genetic.py
├── vendor/aima/            # Vendored AIMA search utilities
├── results/
│   ├── figures/            # Saved plots (PNG)
├── report/
│   └── REPORT.pdf          # Final written report
├── Makefile                # Convenience commands
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Algorithms

### Recommended: Using the Makefile

All algorithms can be run using `make` commands. The input matrix is controlled by the `MAT` variable in the Makefile.

#### Run individual algorithms:
```bash
make nn        # Nearest Neighbor variants
make hill      # Hill Climbing
make sa        # Simulated Annealing
make ga        # Genetic Algorithm
make astar     # A* with MST heuristic
```

#### Change the input matrix:
Edit the `MAT` variable in the Makefile:
```makefile
MAT=data/40_random_adj_mat_0.txt
```

Or override from the command line:
```bash
make nn MAT=data/13_n_gon_adj_mat.txt
make astar MAT=data/10_random_adj_mat_0.txt
```

### Manual Python Commands (Alternative)

If you prefer to run algorithms directly:

#### Nearest Neighbor variants
```bash
python src/aima_nn_algs.py data/13_n_gon_adj_mat.txt
```

#### Hill Climbing
```bash
python src/aima_hill_climbing.py data/40_random_adj_mat_0.txt 150 2000
```

#### Simulated Annealing
```bash
python src/aima_simulated_annealing.py data/40_random_adj_mat_0.txt 0.995 1.0 50000
```

#### Genetic Algorithm
```bash
python src/aima_genetic.py data/40_random_adj_mat_0.txt 0.10 120 400
```

#### A* (with MST heuristic)
```bash
PYTHONPATH=vendor/aima:. python -m src.aima_my_tsp data/10_random_adj_mat_0.txt
```

---

## Important Note on A*

A* reliably finds **optimal solutions** but becomes computationally infeasible beyond **~15 cities** (20+ becomes very slow).

- Node expansion grows rapidly with problem size
- Use smaller matrices when running A*
- Included primarily as a gold-standard reference, not a scalable solver

---

## Figures & Experimental Results

All plots are stored in: `results/figures/`

### Key Figure Categories

1. **Cost vs number of cities**
2. **Runtime and CPU time vs number of cities**
3. **Algorithm / A\* ratios** (cost, runtime, CPU time)
4. **A\* nodes expanded vs problem size**
5. **Hyperparameter sweeps:**
   - RRNN (`k`, `num_repeats`)
   - Hill Climbing (restarts)
   - Simulated Annealing (α, t₀)
   - Genetic Algorithm (mutation rate, population size)
6. **Learning curves** (best-so-far cost vs iteration/generation)

All figures report median values across multiple random matrices, with IQR shading where applicable.

---

## Report

The full written analysis, discussion, and interpretation of results can be found here:

```
report/REPORT.pdf
```

The report addresses:
- Algorithm correctness and optimality
- Hyperparameter tradeoffs
- Scalability limits
- Runtime vs solution quality
- Comparison against A*
- Practical applicability of each algorithm

---

## Author

**Akhil Karumuri**  
University of Maryland

---

## Acknowledgments

- CMSC 421 course staff
- AIMA Python reference implementation
