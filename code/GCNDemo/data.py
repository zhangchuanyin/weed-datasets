'''
Created on 2018-11-14

@author: 南城
'''
import pandas as pd
import numpy as np
import random
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data():
    features=pd.read_excel('feature-xbnll.xlsx')
    labels=pd.read_excel('labels-flower.xlsx')
    labels=np.array(labels)
    adj=pd.read_excel('adj.xlsx')
    features=np.array(features)
    labels=np.array(labels)#标签矩阵
    features=features[:,:]#特征矩阵
    return adj,features,labels
def loadData():
    adj,features,labels=load_data()
    test_idx_reorder=[i for i in range(1500,3670)]
    random.shuffle(test_idx_reorder)   
    test_idx_range = np.sort(test_idx_reorder) 
    features[test_idx_reorder, :] = features[test_idx_range, :]        
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range
    idx_train = range(0,1500)
    train_mask = sample_mask(idx_train, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train,y_test, train_mask,test_mask