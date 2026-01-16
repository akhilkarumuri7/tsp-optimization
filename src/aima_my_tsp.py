"""
Code skeleton for A* with TSP
For use with the AIMA codebase: https://github.com/aimacode/aima-python
CMSC 421 - Fall 2025
"""
from time import time
import numpy as np
import random
from vendor.aima.search import Problem, astar_search
from scipy.sparse import csgraph

### Define TSP ###

class MyTSP(Problem):

    # NOTE: This is just a suggestion for setting up your __init__,
    # you can use any design you want
    def __init__(self, weights):   
        # randomly select a starting city
        start = random.randrange(weights.shape[0])
        initial = (start,) 
        super().__init__(initial, None)
        self.weights = weights # adjacency matrix of weights
        self.num_cities = weights.shape[0] # number of cities
        self.cities = list(range(0, self.num_cities)) # list of cities

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        
        # the state is a tuple of cities visited so far
        # convert it to a set to check which cities have not been visited
        visited = set(state)
        
        # if all cities have been visited, only action is to return to the start
        if len(visited) == self.num_cities:
            return [state[0]] # this returns the first city in the tuple, which is the start city, as a list
        
        # return all the cities that have not been visited yet
        action = []
        for city in self.cities:
            if city not in visited:
                action.append(city)
        return action
                    

    # NOTE: If you make your state a list object, you'll wind
    # up with an error like this: TypeError: unhashable type 'list'
    # One work-around is the make your states tuples instead.
    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        # append chosen city to the path (state is immutable tuple)
        result = state + (action,)
        return result

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        # check if all cities have been visited and we are back at the start
        if len(state) == self.num_cities + 1 and state[0] == state[-1]:
            return True
        else:
            return False
        
        
    # NOTE: Remember the full cost includes the round trip back to the starting city!
    # So if you are adding the final city to the path, you should also add the cost
    # for the final edge too.
    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        
        # returning to the start city is also an action, so we do not need a special case for it.
        # it is handled in the actions() method.
        weight = self.weights[state1[-1], action]
        return c + weight

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        # compute the total cost of the path represented by state
        if len(state) < 2:
            return 0.0
        cost = 0.0
        for i in range(len(state) - 1):
            cost += float(self.weights[state[i], state[i+1]])
        return cost

    # NOTE: For debugging purposes, you can use h(n)=0 while writing and testing
    # the rest of your code
    def h(self, node):
        """Return the heuristic value for a given state. astar_search will
        look for this heuristic function when run."""
        state = node.state
        start = state[0]
        current = state[-1]
        
        # track unvisited cities
        visited = set(state)
        unvisited = []
        for city in self.cities:
            if city not in visited:
                unvisited.append(city)

        # if no unvisited cities remain, heuristic = cost to return home
        if not unvisited:
            return self.weights[current, start]
        
        w = self.weights
        
        # finding the cheapest edge from current city to any unvisited city
        min_cur_to_R = float("inf")
        for r in unvisited:
            cost = w[current, r]
            if cost < min_cur_to_R:
                min_cur_to_R = cost

        # making the submatrix for unvisited cities and finding its MST cost
        subW = w[np.ix_(unvisited, unvisited)]

        mst = csgraph.minimum_spanning_tree(subW)
        mst_cost = float(mst.sum())


        # finding the cheapest edge from any unvisited city to the start city
        min_R_to_start = float("inf")
        for r in unvisited:
            cost = w[r, start]
            if cost < min_R_to_start:
                min_R_to_start = cost

        return min_cur_to_R + mst_cost + min_R_to_start


### Run A* ###

if __name__ == "__main__":
    import sys, time

    if len(sys.argv) < 2:
        print("Usage: python aima_my_tsp.py <matrix_file>")
        sys.exit(1)

    matrix_file = sys.argv[1]
    MAT = np.loadtxt(matrix_file)

    MTSP = MyTSP(MAT)

    start_wall = time.time_ns()
    start_cpu  = time.process_time_ns()
    node = astar_search(MTSP, display=True)
    end_wall = time.time_ns()
    end_cpu  = time.process_time_ns()

    print("\n===== A* (TSP with MST heuristic) =====")
    if node is None:
        print("  No solution found.")
    else:
        path = tuple(node.state)
        cost = MTSP.value(path)
        print("  Path:", path)
        print("  Cost:", cost)

    print("  Wall time (ns):", end_wall - start_wall)
    print("  CPU  time (ns):", end_cpu  - start_cpu)