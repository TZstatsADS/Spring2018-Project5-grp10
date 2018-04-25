#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 18:35:38 2018

@author: yuexuanhuang
"""
import numpy as np
import pandas as pd
import skimage
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.feature import hog
from skimage.feature import daisy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# import train_label, train_image, test_image
label999 = pd.read_csv('/Users/yuexuanhuang/Documents/GitHub/Spring2018-Project5-grp_10/output/GBM/train_label/label999.csv').iloc[:,1:]
train_color = Image.open('/Users/yuexuanhuang/Documents/GitHub/Spring2018-Project5-grp_10/data/GBM/data-ach/7.jpg','r')
label = label999
test_color = Image.open('/Users/yuexuanhuang/Documents/GitHub/Spring2018-Project5-grp_10/data/GBM/Archive/img_1.png','r')

# train model, predict, output
def colorme(train_color, train_label, test_color):

    train_color = train_color.convert('L') #makes it greyscale
    train_gray = np.asarray(train_color.getdata(),dtype=np.float64).reshape((train_color.size[1],train_color.size[0]))
    train_gray = np.asarray(train_gray,dtype=np.uint8) #if values still in range 0-255! 
    train_gray = Image.fromarray(train_gray,mode='L')
    # train_gray.save('/Users/yuexuanhuang/Desktop/Proj_5/alpha2/gray/gray%s.png'%000)
    
    test_color = test_color.convert('L') #makes it greyscale
    test_gray = np.asarray(test_color.getdata(),dtype=np.float64).reshape((test_color.size[1],test_color.size[0]))
    test_gray = np.asarray(test_gray,dtype=np.uint8) #if values still in range 0-255! 
    test_gray = Image.fromarray(test_gray,mode='L')
    # test_gray.save('/Users/yuexuanhuang/Desktop/Proj_5/alpha2/gray/gray%s.png'%001)
    
    # hog(256,256)
    fd, hog_image = hog(train_gray, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)
    
    # daisy(256,256,3)
    descs, descs_img = daisy(train_gray, step=180, radius=58, rings=2, histograms=6,
                             orientations=8, visualize=True)
    
    train = pd.DataFrame(np.matrix(np.full((65536, 5), np.inf)))
    train.columns = ['label', 'hog', 'daisy1', 'daisy2', 'daisy3']
    for i in range(5):
        if i == 0:
            collect = train_label
            for j in range(256):
                for k in range(256):
                    train.iloc[k+j*256, i] = collect.iloc[k,j]
        
        if i == 1:
            collect = np.matrix(hog_image)
            for j in range(256):
                for k in range(256):
                    train.iloc[k+j*256, i] = collect[k,j]
                    
        if i == 2:
            collect = np.matrix(descs_img[:,:,0])
            for j in range(256):
                for k in range(256):
                    train.iloc[k+j*256, i] = collect[k,j]
                    
        if i == 3:
            collect = np.matrix(descs_img[:,:,1])
            for j in range(256):
                for k in range(256):
                    train.iloc[k+j*256, i] = collect[k,j]
                    
        if i == 4:
            collect = np.matrix(descs_img[:,:,2])
            for j in range(256):
                for k in range(256):
                    train.iloc[k+j*256, i] = collect[k,j]
                    
    train_features, test_features, train_labels, test_labels = train_test_split(train.iloc[:,1:], train.iloc[:,0], test_size = 0.3, random_state = 42)
    clf = GradientBoostingClassifier()
    clf.fit(train_features, train_labels)
    # label_predict = clf.predict(test_features)
    # np.mean(label_predict == test_labels)

    # test
    # hog(256,256)
    fd, hog_image = hog(test_gray, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)
    
    # daisy(256,256,3)
    descs, descs_img = daisy(test_gray, step=180, radius=58, rings=2, histograms=6,
                             orientations=8, visualize=True)
    
    test = pd.DataFrame(np.matrix(np.full((65536, 4), np.inf)))
    test.columns = ['hog', 'daisy1', 'daisy2', 'daisy3']
    for i in range(4):
        if i == 0:
            collect = np.matrix(hog_image)
            for j in range(256):
                for k in range(256):
                    test.iloc[k+j*256, i] = collect[k,j]
                    
        if i == 1:
            collect = np.matrix(descs_img[:,:,0])
            for j in range(256):
                for k in range(256):
                    test.iloc[k+j*256, i] = collect[k,j]
                    
        if i == 2:
            collect = np.matrix(descs_img[:,:,1])
            for j in range(256):
                for k in range(256):
                    test.iloc[k+j*256, i] = collect[k,j]
                    
        if i == 3:
            collect = np.matrix(descs_img[:,:,2])
            for j in range(256):
                for k in range(256):
                    test.iloc[k+j*256, i] = collect[k,j]
    
    test_predict = clf.predict(test)
    test_label = pd.DataFrame(np.matrix(np.full((256, 256), np.inf)))
    for i in range(256):
        for j in range(256):
            test_label.iloc[j,i] = test_predict[j+i*256]
    
    return test_label

# get the test_label
t_label = colorme(train_color = train_color, train_label = label, test_color = test_color)

# import the image we need to colorize
test_color = test_color.convert('L') #makes it greyscale
test_gray = np.asarray(test_color.getdata(),dtype=np.float64).reshape((test_color.size[1],test_color.size[0]))
test_gray = np.asarray(test_gray,dtype=np.uint8) #if values still in range 0-255! 
test_gray = Image.fromarray(test_gray,mode='L')
test_gray.save('/Users/yuexuanhuang/Documents/GitHub/Spring2018-Project5-grp_10/output/GBM/gray/gray%s.png'%11)

img = cv2.imread('/Users/yuexuanhuang/Documents/GitHub/Spring2018-Project5-grp_10/data/GBM/Archive/img_1.png')
# img = cv2.resize(img, (256, 256)) 
l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
Lab_l = np.matrix(l)

# define colorize function
def show(standard, t_label, Lab_l):
    Lab_a = np.matrix(np.full((256, 256), np.inf))
    Lab_b = np.matrix(np.full((256, 256), np.inf))
    for i in range(256):
        for j in range(256):
            Lab_a[i,j] = standard[np.int(t_label.iloc[i,j])][0]
            Lab_b[i,j] = standard[np.int(t_label.iloc[i,j])][1]
            
    ar = np.zeros((256,256,3))
    ar[:,:,0] = Lab_l / 2.55
    ar[:,:,1] = Lab_a
    ar[:,:,2] = Lab_b
    rgb = skimage.color.lab2rgb(ar)
    return rgb

# show the colorful picture
plt.imshow(show(standard, t_label, Lab_l))


