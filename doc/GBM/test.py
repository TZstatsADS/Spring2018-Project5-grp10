#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:55:15 2018

@author: yuexuanhuang
"""

# test
# import the gratscale image and colorize them
for i in range(13):
    test_color = Image.open('/Users/yuexuanhuang/Documents/GitHub/Spring2018-Project5-grp_10/data/GBM/Archive/img_%s.png'%(i+1),'r')

    def test(test_color):
        test_color = test_color.convert('L') #makes it greyscale
        test_gray = np.asarray(test_color.getdata(),dtype=np.float64).reshape((test_color.size[1],test_color.size[0]))
        test_gray = np.asarray(test_gray,dtype=np.uint8) #if values still in range 0-255! 
        test_gray = Image.fromarray(test_gray,mode='L')
        
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
        t_label = pd.DataFrame(np.matrix(np.full((256, 256), np.inf)))
        for i in range(256):
            for j in range(256):
                t_label.iloc[j,i] = test_predict[j+i*256]
        return t_label
    
    test_label = test(test_color)
    
    img = cv2.imread('/Users/yuexuanhuang/Documents/GitHub/Spring2018-Project5-grp_10/data/GBM/Archive/img_%s.png'%(i+1))
    # img = cv2.resize(img, (256, 256)) 
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
    Lab_l = np.matrix(l)
    
    result = show(standard, test_label, Lab_l)
    plt.imsave('/Users/yuexuanhuang/Documents/GitHub/Spring2018-Project5-grp_10/output/GBM/movie/result%s.png'%i, result)


