def colorize(img, colors_present,svm,skip=4):
    '''
    -- colorizes a grayscale image, using the set of SVMs defined by train().
    Returns:
    -- ndarray(m,n,3): a mxn pixel RGB image
    '''
       
    m,n = img.shape
    
    num_classified = 0
    
    _,raw_output_a,raw_output_b = cv2.split(cv2.cvtColor(cv2.merge((img, img, img)), 
                                                         cv2.COLOR_RGB2LAB))
    
    output_a = np.zeros(raw_output_a.shape)
    output_b = np.zeros(raw_output_b.shape)
    
    num_classes = len(colors_present)
    label_costs = np.zeros((m,n,num_classes))
    
    g = np.zeros(raw_output_a.shape)
        
    count=0
    scaler = preprocessing.MinMaxScaler() 
    for x in range(0,n,skip):
        for y in range(0,m,skip):
            feat= scaler.transform(get_features(img, (x,y)))
             
            feat=pca.transform(scaler.transform(get_features(img, (x,y))))
     
    feat =   
    feat = 
    count += 1
            
    for i in range(num_classes):
        cost = -1*svm[colors_present[i]].decision_function(feat)[0]
        label_costs[y-int(skip/2):y+int(skip/2)+1,x-int(skip/2):x+int(skip/2)+1,i] = cost

        #edges = self.get_edges(img)
        #self.g = np.sqrt(edges[0]**2 + edges[1]**2)
        self.g = self.get_edges(img)
        #self.g = np.log10(self.g)
      
        if SAVE_OUTPUTS:
            #dump to pickle
            print('saving to dump.dat')
            fid = open('dump.dat', 'wb') 
            pickle.dump({'S': label_costs, 'g': self.g, 'cmap': self.label_to_color_map, 'colors': self.colors_present}, fid)
            fid.close()

        #postprocess using graphcut optimization 
        output_labels = self.graphcut(label_costs, l=self.graphcut_lambda)
        
        for i in range(m):
            for j in range(n):
                a,b = self.label_to_color_map[self.colors_present[output_labels[i,j]]]
                output_a[i,j] = a
                output_b[i,j] = b

        output_img = cv2.cvtColor(cv2.merge((img, np.uint8(output_a), np.uint8(output_b))), cv.CV_Lab2RGB)

        return output_img, self.g
