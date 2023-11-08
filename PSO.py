import random as rand
import numpy as np
import heapq
import matplotlib.pyplot as plt

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
    


pso = PSO(10, 1, 2, 3)
# Assumption: Initial 8 D dimension data
p_size = 3
initial_list = []
w = 0.5
c1 = 1
c2 = 1
vel_list = [round(rand.uniform(-0.5, 0.5), 4) for i in range(p_size)]
for i in range(p_size):
    x_arr = pso.initial_arr()
    initial_list.append(x_arr)
initial_arr = np.array(initial_list)
p_best_fit = [pso.function(x) for x in initial_arr] # 用來存放所有人的fit值
p_best_x_arr = initial_arr # 因為是第一次所以先將原始的初始解放入
g_best_x_arr = initial_arr[p_best_fit.index(min(p_best_fit))] # 這p_size個裡面最棒的array
g_best_fit = p_best_fit # 這群人最小的值
iter_g_best_fit = [] # 用來存放每次迭代的最好值

### 下面要改成goodnotes畫的這種流程圖
for num in range(10):
    print(f'第{num}次迭代:')
    for v, x_index, x_value in zip(vel_list, range(len(initial_arr)), initial_arr):
        print(f'原始值:{x_value}')
        r1 = rand.random()
        r2 = rand.random()
        new_v = (w * v) + (c1 * r1 * (p_best_x_arr[x_index] - x_value)) + (c2 * r2 * (g_best_x_arr - x_value))
        new_x = x_value + new_v
        for index, value in enumerate(new_x):
            if value < -10:
                new_x[index] = -10
            elif value > 10:
                new_x[index] = 10
            else:
                pass 
        print(f'新值:{new_x}')
        if pso.function(new_x) < p_best_fit[x_index]:
            p_best_fit[x_index] = pso.function(new_x)
            p_best_x_arr[x_index] = x_value

        if p_best_fit < g_best_fit:
            g_best_fit = p_best_fit
    iter_g_best_fit.append(g_best_fit)

plt.plot(iter_g_best_fit)
plt.show()

print(f'Global minimum:{g_best_fit}')
