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