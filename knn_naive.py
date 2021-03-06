#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
k-Nearest Neighbors (kNN)

An interface for kNN adapted to distance metric learning algorithms.
"""

from __future__ import absolute_import
import numpy as np
from sklearn import neighbors
from scipy.linalg import sqrtm
from scipy.spatial import distance
from sklearn.neighbors import DistanceMetric
import heapq

class kNN:
    """
        k-Nearest Neighbors (kNN)
        The nearest neighbors classifier adapted to be used with distance metric learning algorithms.

        Parameters
        ----------

        n_neighbors : int

            Number of neighbors to consider in classification.

        dml_algorithm : DML_Algorithm

            The distance metric learning algorithm that will provide the distance in kNN.
    """

    def __init__(self, n_neighbors, dml_algorithm):
        self.nn_ = n_neighbors
        self.dml = dml_algorithm
#        self.knn = neighbors.KNeighborsClassifier(n_neighbors)
#        self.knn_orig = neighbors.KNeighborsClassifier(n_neighbors)

    def distance(self,a,b,type,vi=None):
        if type == 'euclidean':
            return distance.euclidean(a, b)
        elif type == 'manhattan':
            return distance.cityblock(a, b)
        elif type == 'chebyshev':
            return distance.chebyshev(a, b)
        elif type == 'cosine':
            return distance.cosine(a, b)
        elif type == 'correlation':
            return distance.correlation(a, b)
        elif type == 'mahalanobis':
            return distance.mahalanobis(a, b, vi)
        elif type == 'minkowski':
            return distance.minkowski(a, b)
        elif type == 'seuclidean':
            return distance.seuclidean(a, b, V)
        elif type == 'sqeuclidean':
            return distance.seuclidean(a, b)
        
    
    def fit(self,x_test,x_train):
        buffer_size = self.nn_
        pred2return = []
        dists2return = []
        for i in range(x_test.shape[0]): #1400
            dists = np.zeros(x_train.shape[0])
            for j in range(x_train.shape[0]): #5328
                if self.dml=="mahalanobis":
                    test_temp=x_test[i].reshape(1,2048)
                    train_temp=x_train[j].reshape(1,2048)
                    temp=np.vstack((test_temp,train_temp))
                    temp_cov=np.cov(temp.T)
                    eigvals,eigvecs=np.linalg.eig(temp_cov)
                    L=np.diag(eigvals)
                    G=np.dot(sqrtm(L),eigvecs.T)
#                    print(temp_cov.shape)
#                    temp_decomp=np.linalg.cholesky(temp_cov)
#                    G=temp_decomp.T
                    delta=x_test[i]-x_train[j]
                    dist= np.dot(np.dot(G, delta).T, np.dot(G, delta))
                    dist= np.sqrt(dist)
                    print(i," ",j)
                elif self.dml=="seuclidean":
                    test_temp=x_test[i].reshape(2048,1)
                    train_temp=x_train[j].reshape(2048,1)
                    temp=np.hstack((test_temp,train_temp))
                    var_temp=np.var(temp,axis=0)
                    print(i," ",j)
#                    print(temp_cov.shape)
#                    temp_decomp=np.linalg.cholesky(temp_cov)
#                    G=temp_decomp.T
                    dist = self.distance(x_test[i], x_train[j], var_temp, self.dml)  
                else: 
                    dist = self.distance(x_test[i], x_train[j],self.dml)  
                dists[j] = dist
            #dists = heapq.nsmallest(buffer_size, dists)
            idx = dists.argsort()[:buffer_size] #indices of N minimum 
            dists_ = dists[idx]
            #pred = np.asarray(pred)
            dists2return.append(dists_)
            pred2return.append(idx)
        #return top N index in gallery and distances  
        return pred2return, dists2return

            
            
        
        