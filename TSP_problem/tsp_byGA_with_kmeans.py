import pandas as pd
import matplotlib.pyplot as plt
import math as m 
from sklearn.cluster import KMeans
import numpy as np
import random as rand
data = pd.read_excel(r"C:\Users\user\OneDrive - yuntech.edu.tw\文件\Python Scripts\Simulated_Annealing\TSP_problem\Data.xlsx")
Mings = data.iloc[0].values
data = data.drop(0)
data_np = data.to_numpy()

# Kmeans with one limitaion all routes weight have to below or equal 100
loc = data_np[:,1:]

def kmeans_init_center(loc, k):
    m, n = loc.shape
    results = np.array([])
    cores = np.empty((k, n)) 
    cores = loc[np.random.choice(np.arange(m), k, replace = False)] 
    return cores
    
def kmeans_result(loc,k, cores): 
    m, n = loc.shape
    results = np.array([])
    cores = np.empty((k, n)) 
    cores = loc[np.random.choice(np.arange(m), k, replace = False)] 
    balls_limit = [100]* k
    for xy in loc:
#         print("Now loc:",xy)
        temp_result = []
        for c in cores:
            dis = np.sqrt(((xy[0] - c[0]) ** 2) + ((xy[1] - c[1]) ** 2))
            temp_result.append(dis)
            min_temp_result = temp_result.index(min(temp_result))
#         print(min_temp_result)
        if balls_limit[min_temp_result] >= xy[2]:
            balls_limit[min_temp_result] -= xy[2]
            results = np.append(results, temp_result.index(min(temp_result))).reshape(-1,1)
#             print('rest balls',balls_limit)
#             print('-' * 100)
        else:
            temp_result[min_temp_result] = temp_result[temp_result.index(max(temp_result))] 
            min_temp_result = temp_result.index(min(temp_result))
            while balls_limit[min_temp_result] <= xy[2]:
                temp_result[min_temp_result] = temp_result[temp_result.index(max(temp_result))] 
                min_temp_result = temp_result.index(min(temp_result))
            else:
                balls_limit[min_temp_result] -= xy[2]
                results =np.append(results, [temp_result.index(min(temp_result))]).reshape(-1,1)
    data_kmeans = np.column_stack((loc, results))
    return  data_kmeans

def redefine_center(results, k):
    new_cores = np.array([])
    kk = list(range(0,(k)))
    kk_int = [float(i) for i in kk]
    for i in kk_int:
        loc = results[results[:,3] == i][:,:2]
        new_x = np.mean(loc[:,0])
        new_y = np.mean(loc[:,1])
        new_cores = np.append(new_cores, [new_x, new_y])
    return new_cores.reshape(k,2)

def plot(data_kmeans, cores, i):
    ulabel = np.unique(data_kmeans[:,3])
    color=['orange','limegreen','gray','royalblue', 'purple']
    plt.title(f'{i}th kmeans')
    for i in ulabel:
        plt.scatter(data_kmeans[data_kmeans[:,3]==i][:,0], data_kmeans[data_kmeans[:,3]==i][:,1], 
                c = color[int(i)], label = i)
    plt.scatter(cores[:,0], cores[:,1], c = 'red')
    plt.legend()
    plt.gca().legend(('Class 0','Class 1','Class 2','Class 3', 'Class 4'))
    plt.show()
    
cores = kmeans_init_center(loc, 5)
data_kmeans = kmeans_result(loc, 5, cores)
plot(data_kmeans, cores,1)

for i in range(30): 
    new_cores = redefine_center(data_kmeans, 5)
    data_kmeans = kmeans_result(loc, 5, new_cores)
plot(data_kmeans, new_cores, i)

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
            new_sort_list = rand.sample(index_list,len(specific_k_data))  #產生多個不重複的list
            rand_sample_arr = np.append(rand_sample_arr, new_sort_list)
        return rand_sample_arr.reshape(self.initial_gene, len(specific_k_data))
    
    def fitness_cal(self, specific_k_data, random_sample):
        fitness_list = list() #放置不同序列產生的fitness結果
        for rand_s in random_sample:
            fitness = 0 #fitness起始從0開始
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
    
    def cycle_crossover(self, best_gene_sample): # k 為 kmeans 結果， gene 初始解數量，best_num 選出來交配或突變的數量， p交配機率
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

            # 產生新的 O1 and O2
                O1 = []
                O2 = []
                for p1_c1,p2_c1 in zip(g1,g2):
                    #將cycle1留下
                    if p1_c1 in cycle1:
                        O1.append(p1_c1)
                    if p2_c1 in cycle1:
                        O2.append(p2_c1)
                    else:
                        O1.append(g2[g2.index(p2_c1)]) #把c2內的資料對應到p2的位置放入至o1中
                        O2.append(g1[g1.index(p1_c1)]) #把c2內的資料對應到p1的位置放入至o2中
                new_off_arr = np.row_stack((O1,O2))

                offspring.append(O1)
                offspring.append(O2)

            else:
                offspring.append(g1)
                offspring.append(g2)

        return offspring
    
    def mutation(self, offspring_after_crossover):
        best_gene_index = list(range(0, len(offspring_after_crossover))) #不用減一的原因是因為range本來就只會到前一個!
        offspring = list()
        while len(best_gene_index) > 0:
            g_index = rand.choice(best_gene_index)
            best_gene_index.remove(g_index)
            gene = offspring_after_crossover[g_index]
            if rand.random() <= self.mutation_prob:

                mutation_index = list(range(0, len(gene)))
                m_bp=rand.sample(mutation_index,2) #直接利用交換位置
                gene[m_bp[0]],gene[m_bp[1]]=gene[m_bp[1]],gene[m_bp[0]]

                offspring.append(gene)
            else:
                offspring.append(gene)
        return offspring
import time
start_time = time.time()
k_num = range(0,5)
for k in k_num: 
    ga = GA(k,20,10,0.9,0.1) #K = 1, 初始基因解20, 選擇18個, 交配機率. 突變機率
    fitness_list = []
    inter = 50
    while inter != 0:
        inter -= 1
        specific_k_data = ga.take_out_num(data_kmeans)
        random_sample = ga.random_sample(specific_k_data)
        best_gene_sample = ga.fitness_cal(specific_k_data, random_sample)[:ga.best_num].to_numpy()[:,0]
        offspring_after_crossover = ga.cycle_crossover(best_gene_sample)
        offspring_after_mutation = ga.mutation(offspring_after_crossover)
        best_gene_sample = ga.fitness_cal(specific_k_data, offspring_after_mutation)[:ga.best_num].to_numpy()[:,0]
        fitness = ga.fitness_cal(specific_k_data, offspring_after_mutation)[:ga.best_num].to_numpy()[:,1]
        fitness_list.append(fitness[0])
    min_fitness = min(fitness_list)
    min_fitness_index = fitness_list.index(min_fitness)
    plt.title(f'{k} output')
    plt.plot(fitness_list)
    plt.scatter(min_fitness_index,min_fitness, color = 'r')
    plt.annotate(f'min fitness: {min_fitness}', (min_fitness_index + .05, min_fitness + .05))
    plt.show()
end_time = time.time()
# print()
print('運行時間:', round((end_time - start_time), 4))