# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 16:13:06 2018

@author: Eléonore
"""

from SY32_Project_Tools import *

def precision(prediction, truth):
    vp = 0
    fp = 0
    for i in range(0, len(prediction)):
        if prediction[i] == truth[i]:
            vp = vp + 1
        else:
            if (prediction[i] == 1) & (truth[i] == 0):
                fp = fp + 1
    return vp/(vp+fp)
    

def rappel(prediction, truth):
    vp = 0
    fn = 0
    for i in range(0, len(prediction)):
        if prediction[i] == truth[i]:
            vp = vp + 1
        else:
            if (prediction[i] == 0) & (truth[i] == 1):
                fn = fn + 1
    return vp/(vp+fn)

def scoreF1(prediction, truth):
    p = precision(prediction, truth)
    r = rappel(prediction, truth)
    return 2*(p*r)/(p+r)


#def curb():
    
######
#Test du modèle de classifieur
######
    
def cross_validation(x, y, N):
    
    #x = np.reshape(x, (len(x), 24*24))
    print("début cross validation")

    r = np.zeros(N, dtype = float)

    for i in range(0,N):
        mask = np.zeros(x.shape[0], dtype = bool)
        mask[np.arange(i, mask.size, N)] = True
        clf.fit(x[~mask,:], y[~mask])
        r[i] = np.mean(clf.predict(x[mask]) != y[mask])
    
    error = np.mean(r)*100
    #for i in range(0,N):
    #    error += r[i]
    #error = error * 100 / N
    return error, clf.score(x,y)

def validation_script(pos, neg):    
    train, label = label_concat(pos,neg)
    train_s, label_s = shuffle(train,label)
    return cross_validation(train_s, label_s, 5)   

def cross_validation_script(c):
    clf = svm.LinearSVC(C=c)
    fd_hog_pos = compute_hog("\\label") 
    fd_hog_neg = compute_hog("\\neg") 
    return validation_script(fd_hog_pos, fd_hog_neg)

def cross_validation_sliding_window(images, labels, x, y, N):
        #x = np.reshape(x, (len(x), 24*24))
    print("début cross validation")
    box_width = 32
    box_height = 32
    r = np.zeros(N, dtype = float)
    n=0
    predict = np.zeros(len(images)//5, dtype = float)
    for i in range(0,N):
        mask = np.zeros(x.shape[0], dtype = bool)
        #1/5 des données sont mise à True
        mask[np.arange(i, mask.size, N)] = True
        # on entraine sur 4/5
        clf.fit(x[~mask,:], y[~mask])
        
        mask_image = np.zeros(len(images), dtype = bool)
        mask_image[np.arange(i, mask_image.size, N)] = True
        j = 0
        for img in np.array(images)[mask_image]:
            
            num = img.split(".")
            num = int(num[0])
            label_y = labels[num-1]["y"]
            label_x = labels[num-1]["x"]
            label_width = labels[num-1]["width"]
            label_height = labels[num-1]["height"]
            #print("image numero ", j)
            #print(img)
            #print(labels[numero-1]["name"])
            
            # on test sur 1/5
            image = io.imread(img)
            print(num)
            window_x, window_y, window_width, window_height, window_score = sliding_window(image, num, 32, 2)
            
            intersection = intersect(window_x, window_y, window_width, window_height, label_x, label_y, label_width, label_height)
           
            if intersection > 50:
                print("nouvelle image")
                print(window_x)
                print(window_y)
                print("label")
                print(label_x)
                print(label_y)
                print(intersection)
                predict[j] = 1
                n=n+1
                window = image[window_y:window_y+window_height, window_x:window_x+window_width]
                scipy.misc.imsave(origin_path+'\\results\\positive'+str(num)+".jpg", window)
                
            else:
                predict[j] = 0
                window = image[window_y:window_y+window_height, window_x:window_x+window_width]
                window = color.rgb2gray(window)
                window = resize(window, (box_height, box_width))
                scipy.misc.imsave(origin_path+'\\neg2\\neg2'+str(num)+".jpg", window)
            j = j+1
        r[i] = np.mean(predict != 1) 
       
    error = np.mean(r)*100
#    error = 0
#    for i in range(0,N):
#        error += r[i]
#    error = error * 100 / N
    return error, n

def validation_sliding_window_script(path, pos, neg): 
    labels = get_labels()
    images = get_images(path)
    train, label = label_concat(pos,neg)
   # train, label = shuffle(train,label)
    return cross_validation_sliding_window(images, labels, train, label, 5)  

#error, n = validation_sliding_window_script("\\train", fd_hog_label, fd_hog_neg)