import numpy as np
import random as rand
import time 
import matplotlib.pyplot as plt

program_start = time.time() # 程式開始


class TABU_SEARCH():
    
    def __init__(self, tabu_list_max_len): # 初始化
        self.tabu_list_max_len = tabu_list_max_len

    def function(self, x1, x2): # 定義方程式
        x1_square = x1**2
        x1_10_square = (x1 + 10) **2
        fx = 100*(x2 - 0.01*x1_square + 1) + 0.01 * x1_10_square
        return round(fx,4)

    def new_candidate(self, x1, x2, lr): # 產生鄰近點
        neighbor = []
        # 共產生9個鄰近點
        for i in range(-1,2): 
            for j in range(-1,2): 
                new_x1 = round(max(-15, min(15, x1 + i * lr)),3)
                new_x2 = round(max(-3, min(3, x2 + j * lr)),3)
                neighbor.append([new_x1, new_x2]) # 將其鄰近點全部放置陣列中
        return neighbor
 
# Parameter
tabu_list = list()
y_function_arr = list() # 用來放置繪製converge curve的
tabu_list_max_len = 10
iter_ = 1000  # 迭代次數

tabu = TABU_SEARCH(tabu_list_max_len)

# 產生初始解
x1 = rand.uniform(-15,-5) 
x2 = rand.uniform(-3,3)
Best_solution = tabu.function(x1, x2) # 先計算初始解的適應值
best_candidate = np.array([x1,x2]) # 將初始解認定為暫時最佳解 

for i in range(iter_): # 迭代i次
    neighbor = tabu.new_candidate(best_candidate[0], best_candidate[1], 0.01) # 先產生由初始解附近的鄰近解
    best_temp_fitness = float('inf') # 先將暫時最佳解認定為infinte

    for n in neighbor: 
        W = tabu.function(n[0],n[1]) #將每個鄰近點算W
        if W < best_temp_fitness and (n[0], n[1]) not in tabu_list: # 如果此鄰近點比暫時最佳解還要小 且 此鄰近點尚未被放入tabu list時
            best_candidate = np.array([n[0], n[1]]) 
            best_temp_fitness = W  # 將此值取代暫時最佳解

    if best_temp_fitness < Best_solution: # 若暫時最佳解比之前的最佳解還小，則將其取代
        Best_solution = best_temp_fitness 
        y_function_arr.append(Best_solution) # 將值放入置y_function_arr中
    else:
        y_function_arr.append(Best_solution) # 若暫時最佳解沒有比之前的最佳解還小則將原本的最佳解放入陣列中

    tabu_list.append([best_candidate[0],best_candidate[1]]) # 將點放入tabu_list

    if len(tabu_list) > tabu.tabu_list_max_len: 
        tabu_list.remove(tabu_list[0]) # 如果大於預設的tabu list則刪除第一個進來的值(FIFO)

# 印出結果
print("Minimum value:", min(y_function_arr))
print("(x1*, x2*):", best_candidate)

# 繪製Convergence Curve
plt.figure()
plt.title('Convergence Curve')
plt.plot(y_function_arr, label='Convergence Curve')
plt.xlabel('Iterations')
plt.ylabel('Objective Function Value')
plt.legend()
plt.show()
program_end = time.time()        
print(f'Program run time: {round(program_end - program_start, 3)}(sec)')
