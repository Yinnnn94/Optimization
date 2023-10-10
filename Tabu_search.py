import numpy as np
import random as rand
import time 
import matplotlib.pyplot as plt

program_start = time.time()
class TABU_SEARCH():
    
    def __init__(self, Best_solution, tabu_list_max_len, x1_range, x2_range):
        self.Best_solution = Best_solution
        self.tabu_list_max_len = tabu_list_max_len
        self.x1_range = x1_range
        self.x2_range = x2_range

    def function(self, x1, x2):
        x1_square = x1**2
        x1_10_square = (x1 + 10) **2
        fx = 100*(x2 - 0.01*x1_square + 1) + 0.01 * x1_10_square
        return round(fx,4)

    def decay_rate_sigmoid(self, iter_): #
        r = 1 - (1/(1 + np.exp(iter_)))
        return r
    
    def new_candidate(self, x1, x2): 
        new_x1 = x1 + np.random.randn() * self.decay_rate_sigmoid(iter_)
        new_x2 = x2 + np.random.randn() * self.decay_rate_sigmoid(iter_)
        if (new_x1 not in self.x1_range) or (new_x2 not in self.x2_range): 
            new_x1 = rand.uniform(-15,-5)
            new_x2 = rand.uniform(-3,3)
        return new_x1, new_x2
 
# Parameter
tabu_list = list()
y_function_arr = list()
tabu_list_max_len = 10
generate_data = np.array([])
iter_ = 30  #iteration number
Best_solution = float('inf')
n = 30  # numer of n times to do tweaak
x1_range = range(-15, -5)
x2_range = range(-3, 3)
# Program starts
tabu = TABU_SEARCH(Best_solution, tabu_list_max_len, x1_range, x2_range)

#initial value
x1 = rand.uniform(-15,-5) 
x2 = rand.uniform(-3,3)


for i in range(iter_):
    best_temp_fitness = float('inf')
    w_x1, w_x2 = tabu.new_candidate(x1, x2) # 先產生由初始解變動而成的新點
    generate_data = np.append(generate_data, [w_x1, w_x2])
    for gradient_n in range(n): #gradient n times 
        W = tabu.function(w_x1, w_x2) 
        if W < best_temp_fitness and (w_x1, w_x2) not in tabu_list:
            best_candidate = np.array([w_x1, w_x2])
            best_temp_fitness = W
            
    y_function_arr.append(best_temp_fitness)
    tabu_list.append([w_x1, w_x2])             

    if W < Best_solution:
        Best_solution = W
    if len(tabu_list) > tabu.tabu_list_max_len: # 如果大於預設的tabu list則刪除第一個進來的值(FIFO)
        tabu_list.remove(tabu_list[0])


generate_data = generate_data.reshape(int(len(generate_data)/2), 2)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(17,4))
# ax1
ax1.scatter(generate_data[:,0], generate_data[:,1])
ax1.set_title('Generated data')
ax1.set_xlabel('x1')
ax1.axvline(-15, c = 'red')
ax1.axvline(-5, c = 'red')
ax1.axhline(-3, c = 'red')
ax1.axhline(3, c = 'red')
ax1.set_ylabel('x2', rotation = 0)

# ax2
ax2.set_title('Best solution iteration')
ax2.plot(range(1,len(y_function_arr) + 1),y_function_arr)
ax2.axis(xmin = 1, xmax = len(y_function_arr))
plt.show() 

print(f'(x1, x2) = {best_candidate} ,最小值為:{min(y_function_arr)}')
program_end = time.time()        
print(f'Program run time: {round(program_end - program_start, 3)}(sec)')

