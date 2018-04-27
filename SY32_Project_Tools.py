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

def get_resize_bw_image(name, box_width, box_height):
    image = io.imread(name)
    image = color.rgb2gray(image)
    resize(image, (box_height, box_width))
    return image

def compute_hog(path):
    os.chdir(origin_path+path)
    images = glob.glob("*.jpg")
    
    j = 0
    #fd_hog = np.zeros(shape=(len(images), 6804), dtype=float)
	#image_hog = np.zeros(shape=(len(images), len(io.imread(images[0])), len(io.imread(images[0])[0])), dtype=float)

    fd_hog = np.zeros(shape=(len(images), 324), dtype=float)
    image_hog = np.zeros(shape=(len(images), 32, 32), dtype=float)
    
    for i in images:
        image = get_resize_bw_image(i, 32, 32)
        #image.resize(32,32)
        fd_hog[j], image_hog[j] = feature.hog(image, visualise=True)
        j = j+1
    
    return fd_hog
    
    
def show_mean(path):
    image_float = read_img_float(path)
    
    mean = np.mean(image_float, axis=0)
    
    mean_display = io.imshow(mean)
    plt.show(mean_display)




def label_concat(pos, neg):
    train = np.concatenate((pos, neg), axis=0)
    label = np.zeros(len(pos)+len(neg), dtype = int)
    
    for i in range(0, len(pos)):
        label[i] = 1
    
    for i in range(len(pos), len(neg)):
        label[i] = 0
        
    return train, label
    
 
    
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
        num = img.split(".")
        num = int(num[0])
        window_x, window_y, window_width, window_height, window_score = sliding_window(image, num, 32, 2)
        window = image[window_y:window_y+window_height, window_x:window_x+window_width]
        print(len(image))
        print(len(image[0]))
        print(window_x)
        print(window_y)
        print(window_width)
        print(window_height)
        scipy.misc.imsave(origin_path+'\\results\\positive'+str(num)+".jpg", window)
       
		
#Prends en entrée une image non modifiée (en couleur)
def sliding_window(image_orig, num, size, jump):
    
    image = color.rgb2gray(image_orig)
    image_width = len(image[0])
    image_height = len(image)
    
    box_height = size
    box_width = size
    
    max_size = max(box_height, box_width)
    r = 1
    
    min_length = min(image_width, image_height)
    results = np.zeros(shape=(10, 5), dtype = float)   
    nb_results = 0
    
    x = -100
    y = -100
    
    #On itère jusqu'à trouver un visage en changeant la taille de l'image d'origine
    #On pourrait comparer plusieurs tailles d'image
    while (x == -100):
        top = 0
        left = 0
        
        r = r + 0.5
        ratio = (max_size*r)/min_length
        #Amélioration : ici si on ne trouve pas de visage, il faut essayer avec un autre ratio
        
        image_resize = resize(image, (int(image_height*ratio), int(image_width*ratio)))
       

        for i in range(0, (len(image_resize)-box_height)//jump):
            for j in range(0, (len(image_resize[0])-box_width)//jump):
                
                #io.imshow(box)
                box = image_resize[top:top+box_height, left:left+box_width]
                image_hog = feature.hog(box)
                #clf = svm.LinearSVC(0.01)
                if clf.predict(image_hog.reshape(1,-1)):
                    #print("decision")
                    new_score = clf.decision_function(image_hog.reshape(1,-1))
                    if (new_score > 0.1):
                        x = left
                        y = top    
                        
                        window_x = int(x/ratio)
                        window_y = int(y/ratio)
                        window_width = int(box_width/ratio)
                        window_height = int(box_height/ratio)
                        find = False
                        
                        intersection = 0
                        
                        for result in results[:nb_results-1]:
                            #print(result[0])
                            if intersection == 0:
                                intersection = intersect(window_x, window_y, window_width, window_height, int(result[0]), int(result[1]), int(result[2]), int(result[3]))
                                if (intersection > 80):
                                    #il s'agit du même visage
                                    #Faire une vraie fonction de suppression des non-maxima
                                    if (new_score > result[4]):
                                        print("Intersection true", intersection)
                                        result[0] = window_x
                                        result[1] = window_y
                                        result[2] = window_width
                                        result[3] = window_height
                                        result[4] = new_score   
                                    
                        if (intersection == 0):
                            #Trier le tableau par ordre décroissant de score
                            k = 0
                            while (new_score < results[k][4]) & (k < nb_results):
                                k = k + 1
                            #if k < nb_results:
                            results = np.insert(results, k, [window_x, window_y, window_width, window_height, new_score], axis=0)
                            print(results)
                            nb_results = nb_results + 1
                            #else:    
#                                print ("Intersection false", intersection)
#                                results[nb_results][0] = window_x
#                                results[nb_results][1] = window_y
#                                results[nb_results][2] = window_width
#                                results[nb_results][3] = window_height
#                                results[nb_results][4] = new_score

                                
                left = left + jump
            #print("fin ligne")
            left = 0
            top = top + jump
           
    for result in results[:nb_results]:
        file = open(origin_path+"\\label_result.txt", "a")
        file.write(str(num)+" "+str(int(result[0]))+" "+str(int(result[1]))+" "+str(int(result[2]))+" "+str(int(result[3]))+" "+str(result[4])+"\n")
        file.close()
                   
    return int(results[0][0]), int(results[0][1]), int(results[0][2]), int(results[0][3]), results[0][4]

    
def detect_face_script():
    clf = load_model(origin_path+"save_model_001.p")
    return detect_faces("\\test")

