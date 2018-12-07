# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:42:41 2018

@author: zl6415
"""
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.image as mpimg # read images
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from knn_naive import kNN
import time
import sys
import json

# 1467 identities in total
num_identies = 1467
num_validation = 100  
rnd = np.random.RandomState(3)
#
camId = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['camId'].flatten()
filelist = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['filelist'].flatten()
labels = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['labels'].flatten()
train_idx = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten()
#only for testing the design
gallery_idx = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['gallery_idx'].flatten()
query_idx = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['query_idx'].flatten()
with open('PR_data/feature_data.json', 'r') as f:
    features = json.load(f)
features = np.asarray(features) 


def plotimg(filename):
    imgplot = mpimg.imread('PR_data/images_cuhk03/%s' %filename)
    #lena.shape #(512, 512, 3)
    plt.imshow(imgplot)
    
# find distinct identities in training set (should be 767 refer to the protocol)
train_label = labels[train_idx-1,]
iden_train = np.unique(labels[train_idx-1,])
#train_set = np.column_stack((train_idx,labels[train_idx-1,].T))

# use 100 randomly selected identities from training set as validation set
valid_iden = rnd.choice(iden_train, num_validation,replace=False)
valid_index = []
for i in range (num_validation):
    valid_index.append(np.argwhere(train_label == valid_iden[i]))

valid_index = np.concatenate(valid_index, axis=0)
valid_idx = train_idx[valid_index].ravel()
valid_label = labels[valid_idx-1]
train_idx_new = np.delete(train_idx, valid_index)
train_label_new = labels[train_idx_new-1]
features_train = features[train_idx_new-1,:]
features_valid = features[valid_idx-1,:]
features_query = features[query_idx-1,]
features_gallery = features[gallery_idx-1,]
label_query = labels[query_idx-1]
label_gallery = labels[gallery_idx-1]
camId_query = camId[query_idx-1]
camId_gallery = camId[gallery_idx-1]
iden_query = np.unique(label_query)

iden_gallery = np.unique(label_gallery)


#for i in range (gallery_idx.shape[0]):
#    if camId_gallery[i] == 1:
#        features_gallery[i] = 0
#features_gallery_ = features_gallery[~(features_gallery==0).all(1)]  
#label_gallery_ = label_gallery[~(features_gallery==0).all(1)]  
#
#for i in range (query_idx.shape[0]):
#    if camId_query[i] == 2:
#        features_query[i] = 0
#features_query_ = features_query[~(features_query==0).all(1)]
#label_query_ = label_query[~(features_query==0).all(1)]

n_neighbors = 20

#knn classifier with metric defined
clf = kNN(n_neighbors,'seuclidean')
pred, errors = clf.fit(features_query, features_gallery)



# return index in gallery
pred_labels = label_gallery[pred]
for i in range (query_idx.shape[0]):
    for j in range(n_neighbors):
        if (pred_labels[i][j] == label_query[i]) and (camId_query[i] == camId_gallery[pred[i]][j]):
            pred_labels[i][j] = 0

pred_labels_temp = []
N_ranklist = 10 # length of each ranklist
for i in range (query_idx.shape[0]):
    pred_labels_temp.append(pred_labels[i][np.nonzero(pred_labels[i])][:N_ranklist])

#ranklist 
arr_label = np.vstack(pred_labels_temp)
#rank1 accuracy
score_rank1 = accuracy_score(arr_label[:,0], label_query)
#rankk accuracy
rankk=10
#arr_label_rankk=np.zeros(1400,1)
#for i in range(query_idx.shape[0]):
#    for j in range(rankk):
#        if (arr_label[i,j]==label_query[i]):
#            arr_label_rankk[i]=arr_label[i,j]
#            break
#score_rankk = accuracy_score(arr_label_rankk, label_query)
            
    

# =============================================================================
#plotimg(filelist[14065][0])
#release memory
del valid_index


# =============================================================================
# if __name__ == "__main__":
#     preprocessing()
# =============================================================================