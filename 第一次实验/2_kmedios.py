# -*- coding = utf-8 -*-
# @Time : 2023/3/16 19:03
# @Author : 彭睿
# @File : 2_kmedios.py
# @Software : PyCharm
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import random
from sklearn import datasets

plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False


class KMedoids:
    def __init__(self, n_cluster=2, max_iter=10, tol=0.1, start_prob=0.8, end_prob=0.99):
        '''Kmedoids constructor called'''
        if start_prob < 0 or start_prob >= 1 or end_prob < 0 or end_prob >= 1 or start_prob > end_prob:
            raise ValueError('Invalid input')
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.tol = tol
        self.start_prob = start_prob
        self.end_prob = end_prob

        self.medoids = []
        self.clusters = {}
        self.tol_reached = float('inf')
        self.current_distance = 0

        self.__data = None
        self.__is_csr = None
        self.__rows = 0
        self.__columns = 0
        self.cluster_distances = {}

    def fit(self, data):
        self.__data = data
        self.__set_data_type()
        self.__start_algo()
        return self

    def __start_algo(self):
        self.__initialize_medoids()
        self.clusters, self.cluster_distances = self.__calculate_clusters(self.medoids)
        self.__update_clusters()

    def __update_clusters(self):
        for i in range(self.max_iter):
            cluster_dist_with_new_medoids = self.__swap_and_recalculate_clusters()
            if self.__is_new_cluster_dist_small(cluster_dist_with_new_medoids) == True:
                self.clusters, self.cluster_distances = self.__calculate_clusters(self.medoids)
            else:
                break

    def __is_new_cluster_dist_small(self, cluster_dist_with_new_medoids):
        existance_dist = self.calculate_distance_of_clusters()
        new_dist = self.calculate_distance_of_clusters(cluster_dist_with_new_medoids)

        if existance_dist > new_dist and (existance_dist - new_dist) > self.tol:
            self.medoids = cluster_dist_with_new_medoids.keys()
            return True
        return False

    def calculate_distance_of_clusters(self, cluster_dist=None):
        if cluster_dist == None:
            cluster_dist = self.cluster_distances
        dist = 0
        for medoid in cluster_dist.keys():
            dist += cluster_dist[medoid]
        return dist

    def __swap_and_recalculate_clusters(self):
        # http://www.math.le.ac.uk/people/ag153/homepage/KmeansKmedoids/Kmeans_Kmedoids.html
        cluster_dist = {}
        for medoid in self.medoids:
            is_shortest_medoid_found = False
            for data_index in self.clusters[medoid]:
                if data_index != medoid:
                    cluster_list = list(self.clusters[medoid])
                    cluster_list[self.clusters[medoid].index(data_index)] = medoid
                    new_distance = self.calculate_inter_cluster_distance(data_index, cluster_list)
                    if new_distance < self.cluster_distances[medoid]:
                        cluster_dist[data_index] = new_distance
                        is_shortest_medoid_found = True
                        break
            if is_shortest_medoid_found == False:
                cluster_dist[medoid] = self.cluster_distances[medoid]
        return cluster_dist

    def calculate_inter_cluster_distance(self, medoid, cluster_list):
        distance = 0
        for data_index in cluster_list:
            distance += self.__get_distance(medoid, data_index)
        return distance / len(cluster_list)

    def __calculate_clusters(self, medoids):
        clusters = {}
        cluster_distances = {}
        for medoid in medoids:
            clusters[medoid] = []
            cluster_distances[medoid] = 0

        for row in range(self.__rows):
            nearest_medoid, nearest_distance = self.__get_shortest_distance_to_mediod(row, medoids)
            cluster_distances[nearest_medoid] += nearest_distance
            clusters[nearest_medoid].append(row)

        for medoid in medoids:
            cluster_distances[medoid] /= len(clusters[medoid])
        return clusters, cluster_distances

    def __get_shortest_distance_to_mediod(self, row_index, medoids):
        min_distance = float('inf')
        current_medoid = None

        for medoid in medoids:
            current_distance = self.__get_distance(medoid, row_index)
            if current_distance < min_distance:
                min_distance = current_distance
                current_medoid = medoid
        return current_medoid, min_distance

    def __initialize_medoids(self):
        '''Kmeans++ initialisation'''
        self.medoids.append(random.randint(0, self.__rows - 1))
        while len(self.medoids) != self.n_cluster:
            self.medoids.append(self.__find_distant_medoid())

    def __find_distant_medoid(self):
        distances = []
        indices = []
        for row in range(self.__rows):
            indices.append(row)
            distances.append(self.__get_shortest_distance_to_mediod(row, self.medoids)[1])
        distances_index = np.argsort(distances)
        choosen_dist = self.__select_distant_medoid(distances_index)
        return indices[choosen_dist]

    def __select_distant_medoid(self, distances_index):
        start_index = round(self.start_prob * len(distances_index))
        end_index = round(self.end_prob * (len(distances_index) - 1))
        return distances_index[random.randint(start_index, end_index)]

    def __get_distance(self, x1, x2):
        a = self.__data[x1].toarray() if self.__is_csr == True else np.array(self.__data[x1])
        b = self.__data[x2].toarray() if self.__is_csr == True else np.array(self.__data[x2])
        return np.linalg.norm(a - b)

    def __set_data_type(self):
        '''to check whether the given input is of type "list" or "csr" '''
        if isinstance(self.__data, csr_matrix):
            self.__is_csr = True
            self.__rows = self.__data.shape[0]
            self.__columns = self.__data.shape[1]
        elif isinstance(self.__data, list):
            self.__is_csr = False
            self.__rows = len(self.__data)
            self.__columns = len(self.__data[0])
        else:
            raise ValueError('Invalid input')

def plot_graphs(data, k_medoids):
    colors = {0:'b*', 1:'g^',2:'ro',3:'c*', 4:'m^', 5:'yo', 6:'ko', 7:'w*'}
    index = 0
    for key in k_medoids.clusters.keys():
        temp_data = k_medoids.clusters[key]
        x = [data[i][0] for i in temp_data]
        y = [data[i][1] for i in temp_data]
        plt.plot(x, y, colors[index])
        index += 1
    plt.title('簇的构成')
    plt.show()

    medoid_data_points = []
    for m in k_medoids.medoids:
        medoid_data_points.append(data[m])
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    x_ = [i[0] for i in medoid_data_points]
    y_ = [i[1] for i in medoid_data_points]
    plt.plot(x, y, 'yo')
    plt.plot(x_, y_, 'r*')
    plt.title('红点为簇中心点')
    plt.show()

a = [[random.randint(1, 4),random.randint(1, 4)] for i in range(1, 1001)]
n_clusters = range(2,8)
k_medoids = [KMedoids(n_cluster=i) for i in n_clusters]
k_medoids = [k_medoid.fit(a) for k_medoid in k_medoids]
loss = [k_medoid.calculate_distance_of_clusters() for k_medoid in k_medoids]

# 绘制误差曲线以了解最佳的分类簇数
plt.figure(figsize=(13,8))
plt.plot(n_clusters,loss)
plt.xticks(n_clusters)
plt.xlabel('簇数')
plt.ylabel('误差')
plt.title('分簇数与误差情况')
plt.show()

a = [[random.randint(1, 100),random.randint(1, 100)] for i in range(1, 1001)]
k_medoids = KMedoids(n_cluster=3)
k_medoids.fit(a)
plot_graphs(a, k_medoids)

#产生1000个有三个中心点的二维随机样本进行聚类
my_datas = datasets.make_blobs(n_samples=1000,
                               n_features=2,
                               centers=3,
                               center_box = (-10,10),
                               cluster_std=[1.0,2.0,3.0],
                                random_state=2023)
x,y = my_datas
plt.scatter(x[:,0], x[:, 1], c=y, s=8)
xlist=[]
ylist=[]
for i in x[:,0]:
    xlist.append(i)
for i in x[:,1]:
    ylist.append(i)

a = list(zip(xlist,ylist))

n_clusters = range(2,15)
k_medoids = [KMedoids(n_cluster=i) for i in n_clusters]
k_medoids = [k_medoid.fit(a) for k_medoid in k_medoids]
loss = [k_medoid.calculate_distance_of_clusters() for k_medoid in k_medoids]

# 绘制误差曲线以了解最佳的分类簇数
plt.figure(figsize=(13,8))
plt.plot(n_clusters,loss)
plt.xticks(n_clusters)
plt.xlabel('簇数')
plt.ylabel('误差')
plt.title('分簇数与误差情况')
plt.show()

k_medoids = KMedoids(n_cluster=3)
k_medoids.fit(a)
plot_graphs(a, k_medoids)

my_datas= datasets.make_classification(n_samples=1000,
                           n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           n_classes=3,
                           n_clusters_per_class=1,
                           random_state = 2023)
x,y=my_datas

xlist=[]
ylist=[]
for i in x[:,0]:
    xlist.append(i)
for i in x[:,1]:
    ylist.append(i)

a = list(zip(xlist,ylist))

n_clusters = range(2,15)
k_medoids = [KMedoids(n_cluster=i) for i in n_clusters]
k_medoids = [k_medoid.fit(a) for k_medoid in k_medoids]
loss = [k_medoid.calculate_distance_of_clusters() for k_medoid in k_medoids]

# 绘制误差曲线以了解最佳的分类簇数
plt.figure(figsize=(13,8))
plt.plot(n_clusters,loss)
plt.xticks(n_clusters)
plt.xlabel('簇数')
plt.ylabel('误差')
plt.title('分簇数与误差情况')
plt.show()

k_medoids = KMedoids(n_cluster=3)
k_medoids.fit(a)
plot_graphs(a, k_medoids)