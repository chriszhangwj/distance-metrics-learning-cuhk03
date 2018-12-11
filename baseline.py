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
from statistics import mean
from knn_naive import kNN
from mPA_interpolated import mapk, melevenPointAP
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

print("--------finished loading data-----------")
def ranklist(idx_, pred_, label_, camId1, camId2):
    n_neighbors_ = 20
    pred_labels = label_[pred_]
    for i in range (idx_.shape[0]):
        for j in range(n_neighbors_):
            if (pred_labels[i][j] == label_[i]) and (camId1[i] == camId2[pred_[i]][j]):
                pred_labels[i][j] = 0
     
    pred_labels_temp = []
    N_ranklist = 10
    for i in range (idx_.shape[0]):
        pred_labels_temp.append(pred_labels[i][np.nonzero(pred_labels[i])][:N_ranklist])
     
    #ranklist 
    arr_label = np.vstack(pred_labels_temp)
    return arr_label

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

n_neighbors = 5328

#knn classifier with metric defined
clf = kNN(n_neighbors,'chebyshev')
pred, errors = clf.fit(features_query, features_gallery)

print("--------finished KNN-----------")
# return index in gallery
pred_labels = label_gallery[pred]
pred_idx = gallery_idx[pred]

for i in range (query_idx.shape[0]):
    for j in range(n_neighbors):
        if (pred_labels[i][j] == label_query[i]) and (camId_query[i] == camId_gallery[pred[i]][j]):
            pred_labels[i][j] = 0
            pred_idx[i][j] = 0

pred_labels_temp = []
pred_idx_temp = []
N_ranklist = 5320 # length of each ranklist
for i in range (query_idx.shape[0]):
    pred_labels_temp.append(pred_labels[i][np.nonzero(pred_labels[i])][:N_ranklist])
    pred_idx_temp.append(pred_idx[i][np.nonzero(pred_idx[i])][:N_ranklist])

#ranklist 
arr_label = np.vstack(pred_labels_temp)
arr_idx = np.vstack(pred_idx_temp)

print("--------finished compiling rank lists-----------")
#
#-------------------------------------rank1 accuracy-------------------------------------
#score_rank1 = accuracy_score(arr_label[:,0], label_query)

#-------------------------------------rankk accuracy-------------------------------------
#rankk=10
#arr_label_rankk=np.zeros((1400,1))
#for i in range(query_idx.shape[0]):
#    for j in range(rankk):
#        if (arr_label[i,j]==label_query[i]):
#            arr_label_rankk[i]=arr_label[i,j]
#            break
#score_rankk = accuracy_score(arr_label_rankk, label_query)

#---------------------------------rankk accuracy up to k---------------------------------
#rankk=100
#score_rankk=np.zeros([1,rankk])
#for i in range(1,rankk+1): 
#    arr_label_rankk=np.zeros((1400,1))
#    for n in range(query_idx.shape[0]):
#        for m in range(i):
#            if (arr_label[n,m]==label_query[n]):
#                arr_label_rankk[n]=arr_label[n,m]
#                break
#    score_rankk[0,i-1] = accuracy_score(arr_label_rankk, label_query)
    
    
#------------------------r rankk accuracy up to k with CUHK03 protocol-------------------
# note this requires all gallery for KNN
#instance_key=[]
#query_key=[]
#for i in range (query_idx.shape[0]):
#    for j in range (gallery_idx.shape[0]):
#        if (label_query[i] == label_gallery[j]) and (camId_query[i] != camId_gallery[j]):
#            instance_key.append(gallery_idx[j])
#    query_key.append(instance_key)
#    instance_key=[]
#rankk=100
#score_rankk=np.zeros([100,rankk])
#for dice in range (100):
#    for i in range(1,rankk+1):
#        arr_label_rankk=np.zeros((1400,1))
#        for n in range(query_idx.shape[0]):
#            #num_instance=len(query_key[n,:])
#            idx_temp=np.random.choice(query_key[n],1) # randomly select an index
#            for m in range(i):
#                if (arr_idx[n,m]== idx_temp):
#                    arr_label_rankk[n]=arr_label[n,m]
#                    break
#        score_rankk[dice,i-1] = accuracy_score(arr_label_rankk, label_query)
#score_mean_rankk=np.mean(score_rankk, axis=0)


#-------------------------------------mAP at k-------------------------------------------
# first create list of list
actual=label_query.tolist()
type(actual)
type(actual[0])
predicted=arr_label.tolist()
print(type(predicted))
print(type(predicted[0]))
score_map = mapk(actual,predicted,k=N_ranklist)

true_count=np.zeros((1400,1))
for i in range (query_idx.shape[0]):
    count=0
    for j in range(n_neighbors):
        if (pred_labels[i][j] == label_query[i]) and (camId_query[i] != camId_gallery[pred[i]][j]):
            count=count+1
    true_count[i]=count

print("-------- computing melevenPointAP-----------")
##
##-------------------------------interpolated mAP at k------------------------------------
r_list, p_list,inter_precision = melevenPointAP(actual,predicted,true_count)
inter_map_array=[float(sum(col))/len(col) for col in zip(*inter_precision)]



# =============================================================================
#plotimg(filelist[14065][0])
#release memory
del valid_index


# =============================================================================
# if __name__ == "__main__":
#     preprocessing()
# =============================================================================