PY=python

# Fast matrix for A* (exact)
MAT_SMALL=data/10_random_adj_mat_0.txt

# Larger matrix for heuristics (approx)
MAT_LARGE=data/40_random_adj_mat_0.txt

nn:
	$(PY) src/aima_nn_algs.py $(MAT_LARGE)

hill:
	$(PY) src/aima_hill_climbing.py $(MAT_LARGE) 150 2000

sa:
	$(PY) src/aima_simulated_annealing.py $(MAT_LARGE) 0.995 1.0 50000

ga:
	$(PY) src/aima_genetic.py $(MAT_LARGE) 0.10 120 400

astar:
	PYTHONPATH=vendor/aima:. $(PY) -m src.aima_my_tsp $(MAT_SMALL)
