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
from skimage.transform import rescale, resize, downscale_local_mean
import os
import glob

from PIL import Image
from SY32_Project_Data import *
import scipy.misc


import matplotlib.pyplot as plt
clf = svm.LinearSVC()
#origin_path = "C:\\Users\\sy32p009\\Documents\\SY32_PART2\\TD 02 - Classification dimages-20180409\\imageface\\imageface\\"
origin_path = "C:\\Users\\Eléonore\\Documents\\UTC\\GI04\\SY32\\Projet\\SY32_Reconnaissance_Visages"

def read_img_float(path):
    os.chdir(origin_path+path)
    images = glob.glob("*.jpg")
    
    j = 0
    image_float = np.zeros(shape=(len(images), len(io.imread(images[0])), len(io.imread(images[0])[0])), dtype=float)
    for i in images:
        image_float[j]= util.img_as_float(io.imread(i))
        j = j+1
    return image_float
    
#def rgb2gray(rgb):
#	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
def compute_hog(path):
    os.chdir(origin_path+path)
    images = glob.glob("*.jpg")
    
    j = 0
    #fd_hog = np.zeros(shape=(len(images), 6804), dtype=float)
	#image_hog = np.zeros(shape=(len(images), len(io.imread(images[0])), len(io.imread(images[0])[0])), dtype=float)

    fd_hog = np.zeros(shape=(len(images), 324), dtype=float)
    image_hog = np.zeros(shape=(len(images), 32, 32), dtype=float)
    
    for i in images:
        image = io.imread(i)
        image = color.rgb2gray(image)
        #image.resize(32,32)
        fd_hog[j], image_hog[j] = feature.hog(image, visualise=True)
        j = j+1
    
    return fd_hog
    
    
def show_mean(path):
    image_float = read_img_float(path)
    
    mean = np.mean(image_float, axis=0)
    
    mean_display = io.imshow(mean)
    plt.show(mean_display)
    

    
def cross_validation(x, y, N):
    
    #x = np.reshape(x, (len(x), 24*24))
    print("début cross validation")

    r = np.zeros(N, dtype = float)

    for i in range(0,N):
        mask = np.zeros(x.shape[0], dtype = bool)
        mask[np.arange(i, mask.size, N)] = True
        clf.fit(x[~mask,:], y[~mask])
        r[i] = np.mean(clf.predict(x[mask]) != y[mask])
    
    error = 0
    for i in range(0,N):
        error += r[i]
    error = error * 100 / N
    return error

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
    train_s, label_s = label_concat(pos,neg)
    return cross_validation(train_s, label_s, 5)   
    
#    pos_test = read_img_float("test\\pos")
#    neg_test = read_img_float("test\\neg")
#    
#    test = np.concatenate((pos_test, neg_test), axis=0)
#    test = np.reshape(test, (len(test), 24*24))
#    
#    
#    label_test = np.zeros(len(pos_test)+len(neg_test), dtype = int)
#    for i in range(0, len(pos_test)):
#        label_test[i] = 1
#    
#    for i in range(len(pos_test), len(neg_test)):
#        label_test[i] = 0
#    
#    test = np.mean(clf.predict(test) != label_test)
#    return model
    
    
def save_model(clf, file_name):
    s = pickle.dump(clf, open (origin_path+file_name, "wb"))
    
def load_model(file_name):
     my_clf=pickle.load(open(file_name, "rb"))
     return my_clf
 
def detect_faces(path):
    os.chdir(origin_path+path)
    images = glob.glob("*.jpg")
    
    for img in images:
        image = io.imread(img)
        sliding_window(image, img, 32, 2)
       
		
def sliding_window(image, img, size, jump):
    
    image = color.rgb2gray(image)
    
    box_height = size
    box_width = size
    image_width = len(image[0])
    image_height = len(image)
    
    min_length = min(image_width, image_height)
    ratio = (size*2)/min_length
    #Amélioration : ici si on ne trouve pas de visage, il faut essayer avec un autre ratio
    
    image_resize = resize(image, (int(image_height*ratio), int(image_width*ratio)))
    top = 0
    left = 0
    k = 0
    
    width = max(0, (len(image[0])-box_width))
    height = max(0, (len(image)-box_height))
    results = np.zeros(shape=((height//jump)*(width//jump), 3), dtype = int)
        
    
    for i in range(0, (len(image_resize)-box_height)//jump):
        for j in range(0, (len(image_resize[0])-box_width)//jump):
            
            #io.imshow(box)
            box = image_resize[top:top+box_height, left:left+box_width]
            image_hog = feature.hog(box)
            #clf = svm.LinearSVC(0.01)
            if clf.predict(image_hog.reshape(1,-1)):
            #    results[k] = [top, left]
            #    k = k + 1
                intersection =  intersect(results[k-1][1], results[k-1][2], box_width, box_height, left, top, box_width, box_height)
                ### Ici il faudrait regarder quand on a plusieurs carrés qui s'intersectent, il y a plus de chances que ce soit un vrai visage
                ### Donc il faudrait compter le nombre de carrés, et garder le plus central
                if (intersection < 5):
                    print(intersection)
                    results[k][0] = 1
                    results[k][1] = left
                    results[k][2] = top
                    k = k+1
                    box = image_resize[top:top+box_height, left:left+box_width]
                    scipy.misc.imsave(origin_path+'\\results\\positive'+str(img)+".jpg", box)
            left = left + jump
        print("fin ligne")
        left = 0
        top = top + jump
            
    return results


    
def detect_face_script():
    clf = load_model(origin_path+"save_model_001.p")
    return detect_faces("\\test")


def cross_validation_script(c):
    clf = svm.LinearSVC(C=c)
    fd_hog_pos = compute_hog("\\label") 
    fd_hog_neg = compute_hog("\\neg") 

    return validation_script(fd_hog_pos, fd_hog_neg)