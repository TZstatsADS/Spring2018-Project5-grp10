def train_svm(files,ntrain,ncolors,prob,gamma,cost,npca):
    
    #dimensions of image
    m,n = l.shape 
    features = []
    classes = []
    numTrainingExamples = 0
    colors_present = []
    svm = [SVC(probability=prob, gamma=gamma, C=cost) for i in range(ncolors)]
    pca = PCA(npca)
    
    scaler = preprocessing.MinMaxScaler() 
    
    for f in files:

        _,a,b = load_image(f)
        kmap_a = np.concatenate([kmap_a, a.flatten()])
        kmap_b = np.concatenate([kmap_b, b.flatten()])

    color_map=label_to_color_map_fun(kmap_a,kmap_b,ncolors)

    for f in files:
        l,a,b = load_image(f)  
        a,b = self.quantize_kmeans(a,b)
        label=quantize_kmeans(a, b, ncolors)  #get labels for files 
        
        for i in range(ntrain):
        #choose random pixel in training image
            x = int(np.random.uniform(n))
            y = int(np.random.uniform(m))
        
            features.append(get_features(l, (x,y)))
            classes.append(label[x,y])
            numTrainingExamples = numTrainingExamples + 1
        
    # normalize columns
    features =np.array(features)
    classes = np.array(classes)
        
       
    # reduce dimensionality
    #features = pca.fit_transform(features)
    
    for i in range(ncolors):
        if len(np.where(classes==i)[0])>0:
            curr_class = (classes==i).astype(np.int32)
            colors_present.append(i)
            svm[i].fit(features,(classes==i).astype(np.int32))
            
    return colors_present,svm