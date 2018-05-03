# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 19:02:04 2018

@author: arnau
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 19:02:04 2018

@author: arnau
"""

import os
import glob
#from SY32_Project_Path import *

import time 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage import data, color
from skimage import feature
from skimage.transform import rescale, resize, downscale_local_mean

#########


import numpy as np 
from sklearn import svm
from skimage import util
from sklearn.utils import shuffle

import pickle

from skimage import *
import os
import glob

import scipy.misc

origin_path = "C:\\Users\\Eléonore\\Documents\\UTC\\GI04\\SY32\\Projet\\SY32_Reconnaissance_Visages"
########

#Renvoie le numéro d'une image
def get_num(img):
    num = img.split(".")
    num = int(num[0])
    return num

# Renvoe un tableau contenant tous les labels
def get_labels():
    os.chdir(origin_path)
    labels = np.loadtxt("label.txt",dtype={'names': ('name','x', 'y', 'width', 'height'),'formats':('S4','i4','i4','i4','i4')})
    return labels

# Renvoie une liste contenant le nom des images
def get_images(path):
    os.chdir(origin_path+path)
    images = glob.glob("*.jpg")
    return images
    
#Renvoie la taille moyenne d'une box d'après les labels
def get_box_size():
    os.chdir(origin_path)
    labels = np.loadtxt("label.txt",dtype={'names': ('name','x', 'y', 'width', 'height'),'formats':('S4','i4','i4','i4','i4')})
    width = 0
    height = 0
    for label in labels:
        width = width + label["width"]
        height = height + label["height"]
    mean_width = width // len(labels)
    mean_height = height // len(labels)
    
    return mean_width, mean_height
    
    

def color_to_grey(image):
    # on va peut être jouer sur la saturation des couleurs après alors je la met de coté
    return color.rgb2gray(image)
     
# Renvoie le pourcentage d'intersection entre deux boxes
def intersect(x1, y1, w1, h1, x2, y2, w2, h2):

    #Calcul des coordonnées de la box d'intersection
    x3 = max(x1, x2)
    y3 = max(y1, y2)
    
    #print(x3)
    #print(y3)

    #Calcul de la largeur et de la hauteur de la box
    x4 = min(x1+w1, x2+w2)
    y4 = min(y1+h1, y2+h2)
    
    #print(x4)
    #print(y4)

    
    if ((x3 >= x4) | (y3 >= y4)):
        #Pas d'intersection
        return 0
    else:
        
        #Calcul de l'aire de l'intersection
        inter = abs(x4 - x3) * abs(y4 - y3)
        #Calcul de l'aire de l'union des deux boxes
        union = w1 * h1 + w2 * h2 - inter
    		
        #Calcul du pourcentage d'intersection
        ratio = (float(inter)/float(union))*100
    	
        return ratio

#Selectionne des images négatives à partir des labels
# Limit désigne le seuil d'intersection avec le label positif acceptable (0 = pas d'intersection)
# Jump désigne le pas de déplacement de la fenêtre glissante
def get_negative_boxes(image, num, box_width, box_height, jump, limit, label_x, label_y, label_width, label_height):

    # Tableau qui va contenir les images à sauvegarder
    fd_hog = np.zeros(shape=(7, 324), dtype=float)
    
    max_size = max(box_height, box_width)
    r = 0
    
    image_width = len(image[0])
    image_height = len(image)
    
    min_length = min(image_width, image_height)
    
    n=0
    while (n < 7):
        
        #On redimensionne l'image en fonction du ratio (pour avoir des images négatives à différentes
        # échelles )
        left = 0
        top = 0
        r = r + 1
        ratio = (max_size*r)/min_length
        
        image_resize = resize(image, (int(image_height*ratio), int(image_width*ratio)))
        
        #results = np.zeros(shape=((height//jump)*(width//jump), 3), dtype = int)
        for i in range(0, (len(image_resize)-box_height)//jump):
            for j in range(0, (len(image_resize[0])-box_width)//jump):
                
                
                window_x = int(left/ratio)
                window_y = int(top/ratio)
                window_width = int(box_width/ratio)
                window_height = int(box_height/ratio)
                        
                #On calcule l'intersection entre la fenêtre glissante et le label
                intersection = intersect(window_x, window_y, window_width, window_height, label_x, label_y, label_width, label_height)
                #print(intersection)
                
                if ((intersection < limit) & (n < 7)):
                    
                    box = image_resize[top:top+box_height, left:left+box_width]
                    scipy.misc.imsave(origin_path+'\\neg\\negative'+str(num)+'-'+str(n)+".jpg", box)
                    fd_hog[n] = feature.hog(box)
                    n = n + 1
                left = left + jump
            # Fin de la ligne    
            left = 0
            top = top + jump
            
    return fd_hog


# Retourne la box contenant le visage d'après le label, à la taille de notre fenêtre (déformation de l'image)
def get_positive_box(image, num, label_x, label_y, label_width, label_height, box_width, box_height):
     box = image[label_y:label_y+label_height, label_x:label_x+label_width]
     box = resize(box, (box_height, box_width))
     #scipy.misc.imsave(origin_path+'\\label\\box'+str(num)+'.jpg', box) 
     fd_hog = feature.hog(box)
     
     return fd_hog
    
def generate_train_data(path, box_width, box_height, jump, limit):
    
    labels = get_labels()
    images = get_images(path)
    
    p=0 #index du tableau des images positives
    n=0 #index du tableau des images négatives
    fd_hog_pos = np.zeros(shape=(len(images), 324), dtype=float)
    fd_hog_neg = np.zeros(shape=(0, 324), dtype=float)
    for img in images:
        
        image = io.imread(img)
        image = color.rgb2gray(image)
        
        num = get_num(img) 
        
        # On récupère les valeurs associées aux labels dans les fichiers
        label_y = labels[num-1]["y"]
        label_x = labels[num-1]["x"]
        label_width = labels[num-1]["width"]
        label_height = labels[num-1]["height"]        

        # On récupère les box positives, on calcule le hog que l'on sauvegarde dans un tableau, et on sauvegarde aussi les images
        fd_hog_pos[p] = get_positive_box(image, num, label_x, label_y, label_width, label_height, box_width, box_height)
        p = p + 1
        

        # On recherche des images négatives avec une fenêtre glissante dont la taille est adaptée à la taille de l'image
        fd_hog = np.zeros(shape=(7, 324), dtype=float)
        
        fd_hog = get_negative_boxes(image, num, box_width, box_height, jump, limit, label_x, label_y, label_width, label_height)
        fd_hog_neg = np.insert(fd_hog_neg, n, fd_hog, axis=0)
        n = n + 1
        
    return fd_hog_pos, fd_hog_neg


#fd_hog_pos32, fd_hog_neg32 = generate_train_data("\\train", 32, 32, 10, 10)

#32*32 => 324
#32*49 => 648