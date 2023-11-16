from PSO import PSO
from SA import SA
import time 
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import math as m


## PSO program start as here
start_time_PSO = time.time()
# self, D, c1, c2, w, iter_
D = 10
iter_ = 1000
p_size = 50

pso = PSO(D, 1.5, 1, 1, iter_, p_size)
vel_list = pso.generate_vel_intital()
initial_arr = np.array(pso.generate_initial_list())
#產生初始解(p_size x D) <- (3 x 5)
p_best_x_arr = initial_arr # 因為是第一次所以先將原始的初始解放入
p_best_fit = [pso.function(x) for x in initial_arr] # 用來存放所有人的fit值
g_best_x_arr = initial_arr[p_best_fit.index(min(p_best_fit))] # 這p_size個裡面最棒的array(1 x D)
g_best_fit =  pso.function(initial_arr[p_best_fit.index(min(p_best_fit))]) # 這群人最小的值
plot_solution_pso = [] # 用來存放每次迭代的最好值

# Tweak
while pso.iter_ >= 0:
    for v, x_index, x_value in zip(vel_list, range(len(p_best_x_arr)), p_best_x_arr):
        r1 = rand.random()
        r2 = rand.random()
        # 產生新值
        new_v = (pso.w * v) + (pso.c1 * r1 * (p_best_x_arr[x_index] - x_value)) + (pso.c2 * r2 * (g_best_x_arr - x_value))
        new_x = x_value + new_v

        # 用來確認是否新產生的值有超過範圍
        for index, value in enumerate(new_x): 
            if value < -10:
                new_x[index] = -10
            elif value > 10:
                new_x[index] = 10
            else:
                pass 
        # 如果新產出的值計算出的適應值比原本位置得適應值還要小，則取代
        if pso.function(new_x) < p_best_fit[x_index]: 
            p_best_fit[x_index] = pso.function(new_x)
            p_best_x_arr[x_index] = new_x
    
    temp_g_best = pso.function(initial_arr[p_best_fit.index(min(p_best_fit))])
    if temp_g_best < g_best_fit:
        g_best_x_arr = initial_arr[p_best_fit.index(min(p_best_fit))]
        g_best_fit = temp_g_best
    plot_solution_pso.append(g_best_fit)
    pso.iter_ -= 1
end_time_PSO = time.time()
print(f'PSO程式運行時間:{round(end_time_PSO - start_time_PSO, 3)}秒')


## SA program start as here
# t, D, iter_
start_time_SA = time.time()
sa = SA(1000, D, iter_)
x = np.array(sa.initial_data())
best_x = x 
best_solution = sa.function(x)
plot_solution_sa = []

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
        plot_solution_sa.append(best_solution)
    else:
        break
end_time_SA = time.time()
print(f'SA程式運行時間:{round(end_time_SA - start_time_SA, 3)}秒')

print('-' * 100)
print('繪製圖表')
# 繪製圖表
plt.plot(plot_solution_pso, label = 'PSO')
plt.plot(plot_solution_sa, label = 'SA')
plt.xlabel('The number of iteration')
plt.ylabel('Fitness')
plt.legend()
plt.title(f'Converage plot for {D} size data')
plt.show()
print(f'Global minimum for PSO:{g_best_fit}, \n矩陣值為:{g_best_x_arr}')
print(f'最小值出現在:{plot_solution_pso.index(min(plot_solution_pso))}')
print('-'* 100)
print(f'Global minimum for SA:{min(plot_solution_sa)}, \n矩陣值為:{best_x}')
print(f'最小值出現在:{plot_solution_sa.index(min(plot_solution_sa))}')

