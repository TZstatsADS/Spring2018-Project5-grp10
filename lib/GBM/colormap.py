#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 22:11:59 2018

@author: yuexuanhuang
"""
import os
from skimage import io
from sklearn.cluster import KMeans
import numpy as np
import cv2
import pandas as pd
import skimage
import math
global str

# import the color_standard_picture
image = io.imread('/Users/yuexuanhuang/Documents/GitHub/Spring2018-Project5-grp_10/data/GBM/standard/standard.jpg')
image = cv2.resize(image, (256, 256)) 

# define color standard function
def Standard(image, K):
    color = skimage.color.rgb2lab(image)
    l = color[:,:,0]
    a = color[:,:,1]
    b = color[:,:,2]
    
    rows = image.shape[0]
    cols = image.shape[1]
    
    image = image.reshape(rows * cols, 3)
    kmeans = KMeans(n_clusters=K, n_init=2, max_iter=2)
    kmeans.fit(image)
    
    clusters = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)
    labels = np.asarray(kmeans.labels_, dtype=np.uint8)
    labels = labels.reshape(rows, cols)
    
    labels1 = pd.DataFrame(labels)
    a1 = pd.DataFrame(a)
    b1 = pd.DataFrame(b)
    
    c = list()
    for i in range(K):
        A = list()
        B = list()
        for j in range(256):
            for k in range(256):
                if labels1.iloc[j,k] == i:
                    colora = a1.iloc[j,k]
                    colorb = b1.iloc[j,k]
                    A.append(colora)
                    B.append(colorb)
                    
        x = np.mean(A)
        y = np.mean(B)
        c.append([x,y])
        
    return c 

# get the color standard by Kmeans with K = 20
standard = Standard(image, K = 20)

# label the train_image according to our color standard
def Label(standard, image, K):
    def distance(new_x, new_y, x, y):
        d = math.sqrt(np.square((new_x-x)) + np.square((new_y-y)))
        return d
    
    
    labels2 = pd.DataFrame(np.matrix(np.full((256, 256), np.inf)))
    color = skimage.color.rgb2lab(image)
    a2 = color[:,:,1]
    b2 = color[:,:,2]
    a2 = pd.DataFrame(a2)
    b2 = pd.DataFrame(b2)
    
    for i in range(256):
        for j in range(256):
            d = list()
            for k in range(K):
                dis = distance(a2.iloc[i,j], b2.iloc[i,j], standard[k][0], standard[k][1])
                d.append(dis)
            labels2.iloc[i,j] = d.index(min(d))
            
    return labels2

# label all the train_image
str='/Users/yuexuanhuang/Documents/GitHub/Spring2018-Project5-grp_10/data/GBM/data-ach' + '/*.jpg'
coll = io.ImageCollection(str)

path=r'/Users/yuexuanhuang/Documents/GitHub/Spring2018-Project5-grp_10/output/GBM/train_label'
for i in range(len(coll)):
    label  = Label(standard, coll[i], K = 20)
    label.to_csv(os.path.join(path,r'label%s.csv' % i))

