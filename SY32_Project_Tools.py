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
#origin_path = "C:\\Users\\arnau\\Documents\\dev\\P18\\SY32"
hog = 648
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
    image = color_to_grey(image)
    resize(image, (box_height, box_width))
    return image
    
    
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
    
    
def save_model(clf, file_name):
    s = pickle.dump(clf, open (origin_path+file_name, "wb"))
    
def load_model(file_name):
     my_clf=pickle.load(open(file_name, "rb"))
     return my_clf
 
def detect_faces(clf, path, box_width, box_height, min_score, jump):
    images = get_images(path)
    
    # On fait passer la fenêtre glissante sur chaque image
    for img in images:
        image = io.imread(img)
        num = get_num(img)
        sliding_window(clf, image, num, box_width, box_height, min_score, jump)
		
#Prends en entrée une image non modifiée (en couleur)
def sliding_window(clf, image_orig, num, box_width, box_height, min_score, jump):
    
    #On passe l'image en noir et blanc
    image = color_to_grey(image_orig)
    scipy.misc.imsave(origin_path+'\\test_color\\'+str(num)+'.jpg', image)
        
    
    image_width = len(image[0])
    image_height = len(image)
    
    
    max_size = max(box_height, box_width)
    r = 0.6
    
    min_length = min(image_width, image_height)
    
    #Les résultats seront stockés dans un tableau
    results = np.zeros(shape=(0, 5), dtype = float)   
    nb_results = 0
    
    x = -100
    y = -100
    
    #On itère jusqu'à trouver un visage en changeant la taille de l'image d'origine
    #On pourrait comparer plusieurs tailles d'image
    while (x == -100 and r < 3):
        top = 0
        left = 0
        
        #Calcul permettant de tester l'image à différentes échelles
        r = r + 0.4
        ratio = (max_size*r)/min_length
        
        #Redimensionnement de l'image
        image_resize = resize(image, (int(image_height*ratio), int(image_width*ratio)))
       
        for i in range(0, (len(image_resize)-box_height)//jump):
            for j in range(0, (len(image_resize[0])-box_width)//jump):
                
                #On récupère la box                
                box = image_resize[top:top+box_height, left:left+box_width]
                image_hog = feature.hog(box)

                #On teste si la box est un visage
                if clf.predict(image_hog.reshape(1,-1)):
                    
                    #On calcule la certitude de la fonction de décision
                    new_score = clf.decision_function(image_hog.reshape(1,-1))
                    
                    if (new_score > min_score):
                        #Si le score est suffisant, on garde la fenêre
                        x = left
                        y = top    
                        
                        #On remet les proportions d'origine
                        window_x = int(x/ratio)
                        window_y = int(y/ratio)
                        window_width = int(box_width/ratio)
                        window_height = int(box_height/ratio)
                        
                        #Trier le tableau par ordre décroissant de score
                        results = np.insert(results, nb_results, [window_x, window_y, window_width, window_height, new_score], axis=0)
                        nb_results = nb_results + 1
                        #print("Results avant tri", results) 
                        #print("Results après tri", results)

                                
                left = left + jump
            #print("fin ligne")
            left = 0
            top = top + jump
    
    results = results[results[:, 4].argsort()]            
    results = non_maxima(results)

           
    #On enregistre les résultats dans un fichier
    k = 0
    for result in results:
        k = k +1
        window = image[int(result[1]):int(result[1])+int(result[3]), int(result[0]):int(result[0])+int(result[2])]
        scipy.misc.imsave(origin_path+'\\results_s\\positive'+str(num)+"-"+str(k)+".jpg", window)
        file = open(origin_path+"\\label_result.txt", "a")
        file.write(str(num)+" "+str(int(result[0]))+" "+str(int(result[1]))+" "+str(int(result[2]))+" "+str(int(result[3]))+" "+str(result[4])+"\n")
        file.close()
        print("Score final", results[0][4])
                   
    return results

    
def detect_face_script(clf, path, box_width, box_height, min_score, jump):
    return detect_faces(clf, path, box_width, box_height, min_score, jump)

def non_maxima(results): 
    i = 0 
    while i <= (len(results)-2):
        j = i+1  
        continu = 1
        while (j <= (len(results)-1)) & (continu == 1):
            intersection = intersect(results[i][0], results[i][1], results[i][2], results[i][3], results[j][0], results[j][1], results[j][2], results[j][3])
            if intersection > 0:
                print("Resultat supprimé", results[i][4])
                results = np.delete(results, i, axis=0)
                #print("Resultat supprimé")
                continu = 0
            else:
                j = j+1
        if (continu == 1):
            i= i+1
    
    return results