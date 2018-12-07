# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.image as mpimg # read images
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import time
import sys
import json

# 1467 identities in total

num_identies = 1467
num_validation = 100  
rnd = np.random.RandomState(3)

camId = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['camId'].flatten()
filelist = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['filelist'].flatten()
labels = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['labels'].flatten()
train_idx = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten()
# only for testing the design
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

valid_iden = rnd.choice(iden_train, num_validation,replace=False) # random seletion of 100 identities from pool of iden_train (767 identities)
valid_index = []

for i in range (num_validation):
    valid_index.append(np.argwhere(train_label == valid_iden[i])) # gives a numpy array of size 100, each element containing a vector of size 2

# =============================================================================
valid_index = np.concatenate(valid_index, axis=0)
valid_idx = train_idx[valid_index].ravel()
valid_label = labels[valid_idx-1]
train_idx_new = np.delete(train_idx, valid_index)
train_label_new = labels[train_idx_new-1]
features_train = features[train_idx_new-1,:]
features_valid = features[valid_idx-1,:]
 
 
 #classifier = KNeighborsClassifier(n_neighbors=5)  
 #classifier.fit(features_train, train_label_new)  
 #y_pred = classifier.predict(features_valid) 
 
 
 #plotimg(filelist[14065][0])
 #release memory
del valid_index
 
 # =============================================================================
 
 # if __name__ == "__main__":
 
 #     preprocessing()
 
 # =============================================================================
# =============================================================================
