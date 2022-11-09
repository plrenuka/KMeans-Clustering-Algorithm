import numpy as np
import random
import matplotlib.pyplot as plt

def initalize_centroids(feature_data, k):
    random_centroid_indx = random.sample(range(0, len(feature_data)), k)
    centroids = feature_data[random_centroid_indx]
    # print("centroids = ", centroids)
    return centroids

def minkowski_distance(feature_data, query_instance , a):
    distance_sq_list = abs(np.subtract(feature_data, query_instance)) ** a
    minkowski_dist = (np.sum(distance_sq_list, axis = 1)) ** (1/float(a))

    return minkowski_dist

def assign_centroids(feature_data, centroids):
    euclidean_dist_list = []
    assigned_centroids = []
    a = 2
    for i in range(len(centroids)):
        euclidean_dist = minkowski_distance(feature_data, centroids[i], a)
        # distance_sq_list = abs(np.subtract(feature_data, centroids[i])) ** 2
        # euclidean_dist = (np.sum(distance_sq_list, axis=1))**(1/2.0)
        euclidean_dist_list.append(euclidean_dist)
    euclidean_dist_list = np.array(euclidean_dist_list)
    min_distances = np.min(euclidean_dist_list, axis = 0)
    for i in range(len(min_distances)):
        column = euclidean_dist_list[:, i]
        min_column_index = np.where(column == min_distances[i])
        assigned_centroids.append(min_column_index[0][0])
    # print("assigned_centroids = ", assigned_centroids)
    assigned_centroids = np.array(assigned_centroids)
    return assigned_centroids

def move_centroids(feature_data, assigned_centroids, centroids):
    new_centroids = [0 for i in range(len(centroids))]
    v = [0 for i in range(len(centroids))]
    for i in range(0, len(feature_data)):
        c = assigned_centroids[i]
        # print()
        v[c] = v[c] + 1                 #Update per-center counts
        n = 1 / float(v[c])             #get learning rate
        new_centroids[c] = ((1 - n)*c) + (n*(feature_data[i]))  #Take gradient step
    print("new_centroids =", new_centroids)
    new_centroids = np.array(new_centroids)
    return new_centroids

def calculate_cost(feature_data, assigned_centroids, centroids):
    calculated_cost = 0
    for i in range(len(feature_data)):
        dist_square = abs(np.subtract(feature_data[i], centroids[assigned_centroids[i]])) ** 2
        dist_sum = np.sum(dist_square, axis=0)
        calculated_cost = calculated_cost + dist_sum
    # print("calculated_cost , centroids = ", calculated_cost, centroids)
    return calculated_cost

def restart_kmeans(feature_data, k, no_of_iterations, no_of_restarts):
    best_centroids = []
    best_cost = 0
    for i in range(no_of_restarts):
        centroids = initalize_centroids(feature_data, k)
        for j in range(no_of_iterations):
            assigned_centroids = assign_centroids(feature_data, centroids)
            new_centroids = move_centroids(feature_data, assigned_centroids, centroids)
            centroids = new_centroids
        calculated_cost = calculate_cost(feature_data, assigned_centroids, centroids)
        if calculated_cost < best_cost or i == 0:
            best_cost = calculated_cost
            best_centroids = centroids
            # print("updated best_cost =", best_cost)
            # print("updated best_centroids =", best_centroids)
    return best_centroids, best_cost

def elbow_plot(no_of_k, feature_data, no_of_iterations, no_of_restarts):
    values_of_k = list(range(1, no_of_k+1))
    print("values_of_k =", values_of_k)
    cost_values = []
    for k in values_of_k:
        print("k = ", k)
        best_centroids, best_cost = restart_kmeans(feature_data, k, no_of_iterations, no_of_restarts)
        print("cost_values = ", best_cost)
        cost_values.append(best_cost)
    plt.plot(values_of_k, cost_values)
    plt.xticks(values_of_k , rotation='vertical')
    plt.xlabel("k values")
    plt.ylabel("cost")
    plt.show()

def random_batch(feature_data, b):
    random_index = random.sample(range(0, len(feature_data)), b)            #select b random indexes from feature data
    random_data = feature_data[random_index]
    return random_data

def main():
    k = 3
    no_of_iterations = 10
    no_of_restarts = 10
    batch_size = 10000
    feature_data = np.genfromtxt("clusteringData.csv", delimiter=',')
    feature_data = feature_data[: , ]
    # feature_data = feature_data[0:10,:]
    feature_data = random_batch(feature_data, batch_size)
    best_centroids, best_cost = restart_kmeans(feature_data, k, no_of_iterations, no_of_restarts)
    print("final best_cost =", best_cost)
    print("final best_centroids =", best_centroids)

    'uncomment to run elbow plot'
    # no_of_k = 10
    # elbow_plot(no_of_k, feature_data, no_of_iterations, no_of_restarts)

if __name__ == '__main__':
    main()