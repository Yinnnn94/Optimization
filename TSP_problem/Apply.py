from Kmeans import Kmeans_with_limitation
import matplotlib.pyplot as plt
from GA import GA
import pandas as pd
import time

start_time = time.time()
data = pd.read_excel(r"Data.xlsx")
Mings = data.iloc[0].values
data = data.drop(0)
data_np = data.to_numpy()

# Kmeans parameter
k = 5
loc = data_np[:,1:]

# Kmeans Program
kmeans= Kmeans_with_limitation(k, loc)
m, n = kmeans.get_shape()
cores = kmeans.kmeans_init_center(m, n)
data_kmeans = kmeans.kmeans_result(cores, m, n)
for i in range(30): 
    new_cores = kmeans.redefine_center(data_kmeans)
    data_kmeans = kmeans.kmeans_result(new_cores, m, n)

kmeans.plot(data_kmeans, i)

# GA program
k_num = range(0,k)
path_sum = 0
fig, axs = plt.subplots(2, 3, figsize=(20, 5))

for k,ax in zip(k_num, axs.ravel()): 
    ga = GA(k,20, 10, 0.9, 0.1) #K = 1, initial 20 gene, choose 10 gene, cross_probability, mutation_probability
    fitness_list = []
    inter = 50
    while inter != 0:
        inter -= 1
        specific_k_data = ga.take_out_num(data_kmeans)
        random_sample = ga.random_sample(specific_k_data)
        best_gene_sample = ga.fitness_cal(specific_k_data, random_sample, Mings)[:ga.best_num].to_numpy()[:,0]
        offspring_after_crossover = ga.cycle_crossover(best_gene_sample)
        offspring_after_mutation = ga.mutation(offspring_after_crossover)
        best_gene_sample = ga.fitness_cal(specific_k_data, offspring_after_mutation, Mings)[:ga.best_num].to_numpy()[:,0]
        fitness = ga.fitness_cal(specific_k_data, offspring_after_mutation, Mings)[:ga.best_num].to_numpy()[:,1]
        fitness_list.append(fitness[0])
    min_fitness = min(fitness_list)
    path_sum += min_fitness
    min_fitness_index = fitness_list.index(min_fitness)
    ax.set_title(f'{k} output')
    ax.plot(fitness_list)
    ax.scatter(min_fitness_index,min_fitness, color = 'r')
    ax.annotate(f'min fitness: {min_fitness}', (min_fitness_index + .05, min_fitness + .05))
# 刪掉最後一個圖片
axs[1,2].remove()
plt.show()
end_time = time.time()
print(f"Total path:{round(path_sum, 3)}")
print('Runtime:', round((end_time - start_time), 4))