import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

"""
Kladd:

1. Vilkårlige centroids i byrjinga

2. Gruppér etter nærmeste centroid

3. Beregn nye centroids i forhold til grupp(a/ene)

4. Gjenta steg 2 og 3 til maks iterasjoner eller til konvergens
    
"""


class KMeans:
    def __init__(self, k=2, max_iterations=150):
        # Number of clusters
        self.k = k
        
        # Max number of iterations
        self.max_iterations = max_iterations
    
    def fit(self, X):
       
        self.X = X
        # transform panda to numpy
        Xarray = X.to_numpy()
        
        self.rows, self.columns = X.shape
       
        self.centroids = np.empty((self.k, self.columns))
        
        self.clusters = [[] for i in range(self.k)]
        
        # Initialize random centroids
        #Random first centroid
        self.centroids[0] = self.X.sample()

        #K-means ++
        for centroidId in range(1,self.k):
            
            minDistances = []
            for point in Xarray:
                
                distances = []
                for i in range(centroidId):
                    distances.append(euclidean_distance(point, self.centroids[i]))
                minDistances.append(min(distances))
                
            nextCentroid = minDistances.index(max(minDistances))
            self.centroids[centroidId] = Xarray[nextCentroid]
        
           
        # Old implementation of random centroids  
        """
        for i in range(self.k):
            centroid = self.X.sample()
            self.centroids[i] = centroid
        self.printCentroids() 
        """        

        #print(self.centroids)
                
        for q in range(10):  
            counter = 0
            self.clusters = [[] for i in range(self.k)]
            for point in Xarray:
                closestCentroidIndex = self.closestCentroidIndex(point)
                self.clusters[closestCentroidIndex].append(counter)
                counter += 1

            self.getNewCentroids()

            
        # Legg punkta i cluster-lista til nærmeste centroid
        """
        Clusters = liste av lister
            
            Xarray: [0[1,1], 1[2,2], 2[2.7,2.7], 3[2.6,1] 4[2.1,2]]
            closestCentroids: [0,0,1,0,1]
            
            Cluster [
                    Cluster0[0,1,3]
                    Cluster1[2,4]
                    ]
        
        """
        
    def closestCentroidIndex(self, point):
        minDistance = np.inf
        counter = 0
        for centroid in self.centroids:
            distance = euclidean_distance(point, centroid)
            if (minDistance > distance):
                minDistance = distance
                closestCentroidIndex = counter
            counter += 1  
        return closestCentroidIndex
    
    
    def getNewCentroids(self):
        newCentroids = np.zeros((self.k, self.columns))
        count = 0
        for cluster in self.clusters:
            newCentroids[count] = np.mean(self.X.to_numpy()[cluster], axis=0)
            count += 1
        self.centroids = newCentroids
     
    
    def predict(self, X):        
        clusterLabels = []
        for point in X.to_numpy():
            closestCentroidIndex = self.closestCentroidIndex(point)
            clusterLabels.append(closestCentroidIndex)
            
        return clusterLabels    

# Debugging methods      
    def printCentroids(self):
        print("Centroids\n")
        print(self.centroids)
        print("\n")        
    def printClusters(self):
        for i in range(self.k):
            print("cluster")
            print(i)
            print("\nlength")
            print(len(self.clusters[i]))
            print(self.clusters[i])
            print("\n\n")
     
  
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        # TODO: Implement 
        #raise NotImplementedError()
        return self.centroids   
# --- Some utility functions
    
def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 
    
    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.
            
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
