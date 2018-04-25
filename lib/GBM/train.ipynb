#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:45:37 2018

@author: yuexuanhuang
"""

# train the model
# import the label for each train_image
label199 = pd.read_csv('/Users/yuexuanhuang/Documents/GitHub/Spring2018-Project5-grp_10/output/GBM/train_label/label199.csv').iloc[:,1:]
train_color = Image.open('/Users/yuexuanhuang/Documents/GitHub/Spring2018-Project5-grp_10/data/GBM/data-ach/12.jpg','r')
train_label = label199

train_color = train_color.convert('L') #makes it greyscale
train_gray = np.asarray(train_color.getdata(),dtype=np.float64).reshape((train_color.size[1],train_color.size[0]))
train_gray = np.asarray(train_gray,dtype=np.uint8) #if values still in range 0-255! 
train_gray = Image.fromarray(train_gray,mode='L')

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
label_predict = clf.predict(test_features)
np.mean(label_predict == test_labels)
