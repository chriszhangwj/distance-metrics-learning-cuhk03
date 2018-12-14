#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 23:00:25 2018

@author: kevin
"""
import numpy as np

class Rank:
    def __init__(self, n_neighbors):
        self.n_neighbors_ = n_neighbors
    def generate(self, idx_, pred_,label_gallery, label_true, camId1, camId2, gallery_idx):     
        pred_labels = label_gallery[pred_]
        pred_idx = gallery_idx[pred_]
        for i in range (idx_.shape[0]):
            for j in range(self.n_neighbors_):
                if (pred_labels[i][j] == label_true[i]) and (camId1[i] == camId2[pred_[i]][j]):
                    pred_labels[i][j] = 0 
                    pred_idx[i][j] = 0
        pred_labels_temp = []
        pred_idx_temp = []
        N_ranklist = 5320
        for i in range (idx_.shape[0]):
            pred_labels_temp.append(pred_labels[i][np.nonzero(pred_labels[i])][:N_ranklist])
            pred_idx_temp.append(pred_idx[i][np.nonzero(pred_idx[i])][:N_ranklist])
        
        arr_label = np.vstack(pred_labels_temp)
        arr_idx = np.vstack(pred_idx_temp)
        return arr_label, arr_idx
    
