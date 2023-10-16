import pandas as pd
import math as m 
import numpy as np
import random as rand

# GA
class GA:
    def __init__(self, k, initial_gene, best_num, cross_prob, mutation_prob):
        self.k = k  
        self.initial_gene = initial_gene
        self.best_num = best_num        
        self.cross_prob = cross_prob        
        self.mutation_prob = mutation_prob   
    


    def fitness_function(self,x1,x2,y1,y2): # Euclidean distance
        d = m.pow(m.pow(x2 - x1,2) + m.pow(y2 - y1, 2), 0.5)
        return round(d, 4)
    
    def take_out_num(self, data_with_kmeans): 
        specific_k_data = data_with_kmeans[data_with_kmeans[:,3] == self.k]
        return specific_k_data
    
    def random_sample(self, specific_k_data):
        rand_sample_arr = np.array([])
        index_list = list(range(0,len(specific_k_data)))
        for random_times in range(self.initial_gene):  
            new_sort_list = rand.sample(index_list,len(specific_k_data))  
            rand_sample_arr = np.append(rand_sample_arr, new_sort_list)
        return rand_sample_arr.reshape(self.initial_gene, len(specific_k_data))
    
    def fitness_cal(self, specific_k_data, random_sample, Mings):
        fitness_list = list() 
        for rand_s in random_sample:
            fitness = 0 
            for N,N_1 in zip(rand_s, rand_s[1:]):
                fitness += self.fitness_function(specific_k_data[int(N)][0],specific_k_data[int(N_1)][0],specific_k_data[int(N)][1],specific_k_data[int(N_1)][1])
            fitness += self.fitness_function(Mings[1],specific_k_data[int(N_1)][0],Mings[2],specific_k_data[int(N_1)][1]) #最後加上小明家
            if type(rand_s) != list:
                fitness_list.append([rand_s.tolist(), round(fitness,4)])
            else:
                fitness_list.append([rand_s, round(fitness,4)])

        fitness_sample_pd = pd.DataFrame(fitness_list)
        fitness_sample_pd_sorted = fitness_sample_pd.sort_values(by = 1)
        return fitness_sample_pd_sorted
    
    def cycle_crossover(self, best_gene_sample):
        best_gene_index = list(range(0, self.best_num))
        offspring = list()
        while len(best_gene_index) > 0:
            g1_index = rand.choice(best_gene_index)
            best_gene_index.remove(g1_index)
            g2_index = rand.choice(best_gene_index)
            best_gene_index.remove(g2_index)
            g1 = best_gene_sample[g1_index]
            g2 = best_gene_sample[g2_index]
            if rand.random() <= self.cross_prob:
                cycle1 = []
                cycle2 = []
                cycle1.append(g1[0])
                cycle1.append(g2[0])
                for c1 in range(1,len(g1)):
                    if cycle1[0] != cycle1[-1]:
                        cycle1.append(g2[g1.index(cycle1[-1])])

                # 產生第二個cycle 
                for c2 in g1:
                    if c2 not in cycle1 and g1[g1.index(c2)] != g2[g1.index(c2)]:
                        start_num = c2
                        cycle2.append(g1[g1.index(start_num)])
                        cycle2.append(g2[g1.index(start_num)])
                        for c22 in range(1,len(g1)):
                            if cycle2[0] != cycle2[-1]:
                                cycle2.append(g2[(g1.index(cycle2[c22]))]) 
                        break

                O1 = []
                O2 = []
                for p1_c1,p2_c1 in zip(g1,g2):
                    if p1_c1 in cycle1:
                        O1.append(p1_c1)
                    if p2_c1 in cycle1:
                        O2.append(p2_c1)
                    else:
                        O1.append(g2[g2.index(p2_c1)]) 
                        O2.append(g1[g1.index(p1_c1)]) 

                offspring.append(O1)
                offspring.append(O2)

            else:
                offspring.append(g1)
                offspring.append(g2)

        return offspring
    
    def mutation(self, offspring_after_crossover):
        best_gene_index = list(range(0, len(offspring_after_crossover))) 
        offspring = list()
        while len(best_gene_index) > 0:
            g_index = rand.choice(best_gene_index)
            best_gene_index.remove(g_index)
            gene = offspring_after_crossover[g_index]
            if rand.random() <= self.mutation_prob:

                mutation_index = list(range(0, len(gene)))
                m_bp=rand.sample(mutation_index,2) 
                gene[m_bp[0]],gene[m_bp[1]]=gene[m_bp[1]],gene[m_bp[0]]

                offspring.append(gene)
            else:
                offspring.append(gene)
        return offspring

