import random as rand
import numpy as np
import matplotlib.pyplot as plt
import time

class PSO():

    def __init__(self, D, c1, c2, w, iter_):
        self.iter_ = iter_ 
        self.D = D
        self.c1 = c1
        self.c2 = c2
        self.w = w

    def function(self, x_arr):
        cal_x_arr = x_arr * np.pi/180
        x_arr_sin = np.sin(cal_x_arr) 
        return round(sum(abs(x_arr * x_arr_sin + (0.1 * x_arr))), 4)

    def initial_data(self):
        x_list = [round(rand.uniform(-10, 10), 4) for i in range(self.D)]
        return x_list
    
start_time = time.time()
# self, D, c1, c2, w, iter_
pso = PSO(50, 1.5, 1, 1, 1000)
p_size = 10
initial_list = []
vel_list = [round(rand.uniform(-0.5, 0.5), 4) for i in range(p_size)]

# 產生p_size筆資料
for i in range(p_size):
    x_arr = pso.initial_data()
    initial_list.append(x_arr)

#產生初始解(p_size x D) <- (3 x 5)
initial_arr = np.array(initial_list) 
p_best_x_arr = initial_arr # 因為是第一次所以先將原始的初始解放入
p_best_fit = [pso.function(x) for x in initial_arr] # 用來存放所有人的fit值
g_best_x_arr = initial_arr[p_best_fit.index(min(p_best_fit))] # 這p_size個裡面最棒的array(1 x D)
g_best_fit =  pso.function(initial_arr[p_best_fit.index(min(p_best_fit))]) # 這群人最小的值
iter_g_best_fit = [] # 用來存放每次迭代的最好值

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
    iter_g_best_fit.append(g_best_fit)
    pso.iter_ -= 1

# 繪製圖表
plt.plot(iter_g_best_fit)
plt.xlabel('The number of iteration')
plt.ylabel('Fitness')
plt.title('Converage plot for PSO')
plt.show()
print(f'Global minimum:{g_best_fit}, \n矩陣值為:{g_best_x_arr}')
end_time = time.time()
print(f'程式運行時間:{round(end_time - start_time, 3)}秒')