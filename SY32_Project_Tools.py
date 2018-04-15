# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:32:03 2018

@author: sy32p009
"""

import numpy as np 
from sklearn import svm
from skimage import util
from sklearn.utils import shuffle

import pickle

from skimage import io
from skimage import feature
from skimage import color
import os
import glob

from PIL import Image

import matplotlib.pyplot as plt
clf = svm.LinearSVC()
#origin_path = "C:\\Users\\sy32p009\\Documents\\SY32_PART2\\TD 02 - Classification dimages-20180409\\imageface\\imageface\\"
origin_path = "C:\\Users\\sy32p009\\Documents\\SY32_PART2\\TD 02 - Classification dimages-20180413\\imagepers\\"

def read_img_float(path):
    os.chdir(origin_path+path)
    images = glob.glob("*.png")
    
    j = 0
    image_float = np.zeros(shape=(len(images), len(io.imread(images[0])), len(io.imread(images[0])[0])), dtype=float)
    for i in images:
        image_float[j]= util.img_as_float(io.imread(i))
        j = j+1
    return image_float
    
def compute_hog(path):
    os.chdir(origin_path+path)
    images = glob.glob("*.png")
    
    j = 0
    fd_hog = np.zeros(shape=(len(images), 6804), dtype=float)
    image_hog = np.zeros(shape=(len(images), len(io.imread(images[0])), len(io.imread(images[0])[0])), dtype=float)
    
    for i in images:
        image = io.imread(i)
        fd_hog[j], image_hog[j] = feature.hog(image, visualise=True)
        j = j+1
    
    return fd_hog
    
    
def show_mean(path):
    image_float = read_img_float(path)
    
    mean = np.mean(image_float, axis=0)
    
    mean_display = io.imshow(mean)
    plt.show(mean_display)
    

    
def cross_validation(x, y, N):
    
    x = np.reshape(x, (len(x), 24*24))
    r = np.zeros(N, dtype = float)

    for i in range(0,N):
        mask = np.zeros(x.shape[0], dtype = bool)
        mask[np.arange(i, mask.size, N)] = True
        model = clf.fit(x[~mask,:], y[~mask])
        r[i] = np.mean(clf.predict(x[mask]) != y[mask])
    
    error = 0
    for i in range(0,N):
        error += r[i]
    error = error * 100 / N
    print(error)
    return model

def label_concat(pos, neg):
    train = np.concatenate((pos, neg), axis=0)
    label = np.zeros(len(pos)+len(neg), dtype = int)
    
    for i in range(0, len(pos)):
        label[i] = 1
    
    for i in range(len(pos), len(neg)):
        label[i] = 0

    train_s, label_s = shuffle(train,label)
    return train_s, label_s
    
def validation_script(pos, neg):
    train = np.concatenate((pos, neg), axis=0)
    label = np.zeros(len(pos)+len(neg), dtype = int)
    
    for i in range(0, len(pos)):
        label[i] = 1
    
    for i in range(len(pos), len(neg)):
        label[i] = 0
    
    train_s, label_s = shuffle(train,label)


    model = cross_validation(train_s, label_s, 5)   
    
    pos_test = read_img_float("test\\pos")
    neg_test = read_img_float("test\\neg")
    
    test = np.concatenate((pos_test, neg_test), axis=0)
    test = np.reshape(test, (len(test), 24*24))
    
    
    label_test = np.zeros(len(pos_test)+len(neg_test), dtype = int)
    for i in range(0, len(pos_test)):
        label_test[i] = 1
    
    for i in range(len(pos_test), len(neg_test)):
        label_test[i] = 0
    
    test = np.mean(clf.predict(test) != label_test)
    return model
    
    
def save_model(clf):
    s = pickle.dump(clf, open ("save.p", "wb"))
    
def load_model():
     my_clf=pickle.load(open("save.p", "rb"))
     return my_clf
     
def sliding_window(image):
    
    image= color.rgb2gray(image)
    left = 0
    top = 0  
    width = 64
    height = 128
    
    saut = 3 
    #area = image[top:top+height, left:left+width]
    #io.imshow(area)
        
    results = np.zeros(shape=(500, 2), dtype = int)
    k = 0
    
    for i in range(0, (len(image[0])-width)//saut):
        for j in range(0, (len(image)-height)//saut):
            box = image[top:top+height, left:left+width]
            io.imshow(box)
            image_hog = feature.hog(box)
            print(len(image_hog))
            if clf.predict(image_hog):
                results[k] = [top, left]
                k = k + 1
            left = left + saut
        top = top + saut
            
    return image_hog

    
    