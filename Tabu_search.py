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

    def decay_rate_sigmoid(self, iter_): #
        r = 1 - (1/(1 + np.exp(iter_)))
        return r
    
    def new_candidate(self, x1, x2):
        new_x1 = x1 + np.random.randn() * self.decay_rate_sigmoid(iter_)
        new_x2 = x2 + np.random.randn() * self.decay_rate_sigmoid(iter_)
        return new_x1, new_x2




tabu_list = list()
tabu_list_max_len = 20
iter_ = 10 #iteration number
Best_solution = float('inf')

n = 10 # numer of n times to do tweaak
tabu = TABU_SEARCH(Best_solution, tabu_list_max_len)
x1 = rand.uniform(-15,-5)
x2 = rand.uniform(-3,3)
S = tabu.function(x1, x2)
Best_solution = S
while iter_ > 0:
    print(f'第{iter_}迭代:')
    if len(tabu_list) <= tabu_list_max_len:
        r_x1, r_x2 = tabu.new_candidate(x1, x2)
        R = tabu.function(r_x1, r_x2) #transform from initial solution 1 time
        print(f'第一次初始值轉變後結果(R):{R}')
        for i in range(n): #gradient n times 
            trigger_flag = False
            w_x1, w_x2 = tabu.new_candidate(x1, x2) # transform from initial solution n times (20)
            W = tabu.function(w_x1, w_x2) 
            if (W < R) or (R in tabu_list):
                trigger_flag = True 
            if trigger_flag == True and (W not in tabu_list):
                print('要放入喔')
                R = W #R為暫時最小的值
            else:
                print('不放!!!')
            if R not in tabu_list:
                S = R # transform S to R( which R is transformed by W )
                print(f'new S :{S}')
                tabu_list.append(R)
            else:
                print('已經放過了!')
            if S < Best_solution:
                print()
                Best_solution = S
                print(f'Best solution(從S變過來的):{Best_solution}')
            else:
                print(f'Best solution(未改過):{Best_solution}')
        print(f'tabu list 長度: {len(tabu_list)}')  
    else:
        print('刪掉最舊得值中...')
        tabu_list.remove(tabu_list[0])
        print(f'tabu list 長度: {len(tabu_list)}')
    iter_ -= 1


print(tabu_list, len(tabu_list))

    
# else:
#     print(Best_solution)