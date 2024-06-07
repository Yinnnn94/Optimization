import matplotlib.pyplot as plt
import numpy as np

# Kmeans with one limitaion all routes weight have to below or equal 100

class Kmeans_with_limitation():
    def __init__(self, k, loc):
        self.k = k # Cluster number
        self.loc = loc # The numpy array which puts (x, y, weight) 

    def get_shape(self):
        m, n = self.loc.shape
        return m, n        

    def kmeans_init_center(self, m, n):
        cores = np.empty((self.k, n)) 
        cores = self.loc[np.random.choice(np.arange(m), self.k, replace = False)] 
        return cores
        
    def kmeans_result(self, cores, m, n): 
        results = np.array([])
        cores = np.empty((self.k, n)) 
        cores = self.loc[np.random.choice(np.arange(m), self.k, replace = False)] 
        balls_limit = [100]* self.k
        for xy in self.loc:
            temp_result = []
            for c in cores:
                dis = np.sqrt(((xy[0] - c[0]) ** 2) + ((xy[1] - c[1]) ** 2))
                temp_result.append(dis)
                min_temp_result = temp_result.index(min(temp_result))
            if balls_limit[min_temp_result] >= xy[2]:
                balls_limit[min_temp_result] -= xy[2]
                results = np.append(results, temp_result.index(min(temp_result))).reshape(-1,1)

            else:
                temp_result[min_temp_result] = temp_result[temp_result.index(max(temp_result))] 
                min_temp_result = temp_result.index(min(temp_result))
                while balls_limit[min_temp_result] <= xy[2]:
                    temp_result[min_temp_result] = temp_result[temp_result.index(max(temp_result))] 
                    min_temp_result = temp_result.index(min(temp_result))
                else:
                    balls_limit[min_temp_result] -= xy[2]
                    results =np.append(results, [temp_result.index(min(temp_result))]).reshape(-1,1)
        data_kmeans = np.column_stack((self.loc, results))
        return  data_kmeans

    def redefine_center(self, results):
        new_cores = np.array([])
        kk = list(range(0,(self.k)))
        kk_int = [float(i) for i in kk]
        for i in kk_int:
            locs = results[results[:,3] == i][:,:2]
            new_x = np.mean(locs[:,0])
            new_y = np.mean(locs[:,1])
            new_cores = np.append(new_cores, [new_x, new_y])
        return new_cores.reshape(self.k,2)

    def plot(self, data_kmeans, i):
        ulabel = np.unique(data_kmeans[:,3])
        color=['orange','limegreen','gray','royalblue', 'purple']
        plt.title(f'{i}th kmeans')
        for i in ulabel:
            plt.scatter(data_kmeans[data_kmeans[:,3]==i][:,0], data_kmeans[data_kmeans[:,3]==i][:,1], 
                    c = color[int(i)], label = i)
        plt.legend()
        plt.gca().legend(('Class 0','Class 1','Class 2','Class 3', 'Class 4'))
        plt.show()