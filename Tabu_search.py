import numpy as np
import random as rand


class TABU_SEARCH():
    
    def __init__(self, Best_solution, tabu_list_max_len):
        self.Best_solution = Best_solution
        self.tabu_list_max_len = tabu_list_max_len

    def function(self, x1, x2):
        x1_square = x1**2
        x1_10_square = (x1 + 10) **2
        fx = 100*(x2 - 0.01*x1_square + 1) + 0.01* x1_10_square
        return round(fx,4)

    def new_candidate(self, x1, x2):
        new_x1 = rand.uniform(-15,-5)
        new_x2 = rand.uniform(-3,3)
        return new_x1, new_x2

    def gradient_descent(self, x1, x2):
        x1_x2 = np.array([x1, x2])
        x1_x2_gradient = np.gradient(x1_x2)
        return x1_x2_gradient



tabu_list = list()
tabu_list_max_len = 10
iter_ = 100 #iteration number
Best_solution = float('inf')
x1 = rand.uniform(-15,-5)
x2 = rand.uniform(-3,3)

tabu = TABU_SEARCH(Best_solution, tabu_list_max_len)
x1_x2_gradient = tabu.gradient_descent(x1,x2)
print(x1, x2, x1_x2_gradient)