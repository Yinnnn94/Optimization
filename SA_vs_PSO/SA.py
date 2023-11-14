import random as rand
import numpy as np
import matplotlib.pyplot as plt
import math as m
import time

class SA():
    
    def __init__(self, t, D, iter_):
        self.t = t
        self.iter_ = iter_
        self.D = D
        
    def function(self, x):
        cal_x_arr = x * np.pi/180
        x_arr_sin = np.sin(cal_x_arr) 
        return round(sum(abs(x * x_arr_sin + (0.1 * x))), 4)

    def initial_data(self):
        x_list = [round(rand.uniform(-10, 10), 4) for i in range(self.D)]
        return x_list
    
    def generate_neighbor(self, x_list):
        new_x = x_list.copy()
        for i in range(len(x_list)):
            new_x[i] += rand.gauss(0, 0.5)
            new_x[i] = round(max(min(new_x[i], 10), -10), 4)
        return new_x

start_time = time.time()
# t, D, iter_, 
sa = SA(1000, 50, 1000)
x = np.array(sa.initial_data())
best_x = x 
best_solution = sa.function(x)
plot_solution = []

for i in range(sa.iter_):
    if sa.t > 0:
        current_solution = sa.function(x)
        new_x = np.array(sa.generate_neighbor(x))
        temp_solution = sa.function(x)
        if temp_solution < current_solution or rand.uniform(0, 1) < m.exp((current_solution - temp_solution)/sa.t):
            current_solution = temp_solution
            x = new_x
        if current_solution < best_solution:
            best_solution = current_solution
            best_x = x
        sa.t = round(sa.t * 0.95, 4)
        plot_solution.append(best_solution)
    else:
        break

print(f'Global minimum:{min(plot_solution)}, \n矩陣值為:{best_x}')
plt.plot(plot_solution)
plt.xlabel('The number of iteration')
plt.ylabel('Fitness')
plt.title('Converage plot for SA')
plt.show()
end_time = time.time()
print(f'程式運行時間:{round(end_time - start_time, 3)}秒')