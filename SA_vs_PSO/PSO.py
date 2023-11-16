import random as rand
import numpy as np

class PSO():

    def __init__(self, D, c1, c2, w, iter_, p_size):
        self.D = D
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.iter_ = iter_ 
        self.p_size = p_size

    def function(self, x_arr):
        cal_x_arr = x_arr * np.pi/180
        x_arr_sin = np.sin(cal_x_arr) 
        return round(sum(abs(x_arr * x_arr_sin + (0.1 * x_arr))), 4)

    def generate_initial_data(self):
        x_list = [round(rand.uniform(-10, 10), 4) for i in range(self.D)]
        return x_list
    
    def generate_vel_intital(self):
        return [round(rand.uniform(-0.5, 0.5), 4) for i in range(self.p_size)] 

    def generate_initial_list(self):
        initial_list = []
        for i in range(self.p_size):
            initial_list.append(self.generate_initial_data())
        return initial_list