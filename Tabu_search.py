import numpy as np
import random as rand

class TABU_SEARCH():
    
    def __init__(self, Best_solution, tabu_list_max_len):
        self.Best_solution = Best_solution
        self.tabu_list_max_len = tabu_list_max_len

    def function(self, x1, x2):
        x1_square = x1**2
        x1_10_square = (x1 + 10) **2
        fx = 100*(x2 - 0.01*x1_square + 1) + 0.01 * x1_10_square
        return round(fx,4)

    def decay_rate_sigmoid(self, iter_): #
        r = 1 - (1/(1 + np.exp(iter_)))
        return r
    
    def new_candidate(self, x1, x2): #若超出範圍時這時該怎麼做?
        new_x1 = x1 + np.random.randn() * self.decay_rate_sigmoid(iter_)
        new_x2 = x2 + np.random.randn() * self.decay_rate_sigmoid(iter_)
        return new_x1, new_x2
 
# Parameter
tabu_list = list()
y_function_arr = np.array([])
tabu_list_max_len = 10
iter_ = 3  #iteration number
Best_solution = float('inf')
n = 2  # numer of n times to do tweaak

# Program starts
tabu = TABU_SEARCH(Best_solution, tabu_list_max_len)

#initial value
x1 = rand.uniform(-15,-5) 
x2 = rand.uniform(-3,3)


for iter in range(iter_):
    print(f'第{iter}迭代:')
    best_candidate = None #先設為空值
    best_temp_fitness = float('inf')
    w_x1, w_x2 = tabu.new_candidate(best_candidate[0], best_candidate[1]) # 產生新點
    
    for gradient_n in range(n): #gradient n times 
        if (w_x1, w_x2) not in tabu_list:
            W = tabu.function(w_x1, w_x2) 
        if W < best_temp_fitness:
            best_candidate = np.array[w_x1, w_x2]
            best_temp_fitness = W
    if best_candidate == None:
        break


    #     tabu.tabu_list.append(Best_solution)             
        
    #     if S < Best_solution:
    #         Best_solution = S
    #         print(f'Best solution(從S變過來的):{Best_solution}')
    #     else:
    #         print(f'Best solution(未改過):{Best_solution}')   
    
    # if len(tabu_list) > tabu_list_max_len: # 如果大於預設的tabu list則刪除第一個進來的值(FIFO)
    #     print('刪掉最舊的值中...')
    #     tabu_list.remove(tabu_list[0])
    #     print(f'tabu list 長度: {len(tabu_list)}, \ntabu list有{tabu_list}')


# print(f'tabu list 長度: {len(tabu_list)}, \ntabu list有{tabu_list}')
# print(f'(x1, x2) = {best_candidate} ,最小值為:{min(tabu_list)}')

    
# else:
#     print(Best_solution)