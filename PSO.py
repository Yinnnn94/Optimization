import random as rand
import numpy as np

class PSO():

    def __init__(self, D, c1, c2, w):
        self.D = D
        self.c1 = c1
        self.c2 = c2
        self.w = w


    def function(self, x_arr):
        cal_x_arr = x_arr * np.pi/180
        x_arr_sin = np.sin(cal_x_arr) 
        return round(sum(abs(x_arr * x_arr_sin + (0.1 * x_arr))), 4)

    def initial_arr(self):
        x_list = [round(rand.uniform(-10, 10), 4) for i in range(self.D)]
        # x_arr = np.array(x_list)
        return x_list
    
    # def pso(self, x_arr, vel_arr):


pso = PSO(3, 1, 2, 3)
# Assumption: Initial 8 D dimension data
p_size = 3
initial_list = []
w = 1
c1 = 3
c2 = 2
vel_list = [round(rand.uniform(-5, 5), 4) for i in range(p_size)]
for i in range(p_size):
    x_arr = pso.initial_arr()
    initial_list.append(x_arr)
initial_arr = np.array(initial_list)
print(initial_arr)
fitness = [pso.function(x) for x in initial_arr]
this_time_best = initial_arr[fitness.index(min(fitness))]
g_best_x_arr = this_time_best
print(g_best_x_arr, this_time_best)
# for v, x_index, x_value in zip(vel_list, range(len(initial_arr)), initial_arr):
#     print(v, x_value)
#     this_time_best = []
#     new_v_list = []
#     r1 = rand.random()
#     r2 = rand.random()
#     new_v = (w * v) + (c1 * r1 * (this_time_best - x_value)) + (c2 * r2 * (g_best_x_arr - x_value))
#     new_x = x_value + new_v
#     initial_arr[x_index] = new_x
#     print(initial_arr)
# print(p_best_index, initial_arr[p_best_index])