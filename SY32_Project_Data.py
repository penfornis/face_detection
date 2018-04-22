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


def get_box_positives(path):
    os.chdir(origin_path)
    labels = np.loadtxt("label.txt",dtype={'names': ('name','x', 'y', 'width', 'height'),'formats':('S4','i4','i4','i4','i4')})
    os.chdir(origin_path+path)
    images = glob.glob("*.jpg")
    
    box_heigth = 200
    box_width = 200
    fd_hog = np.zeros(shape=(len(images), 42849), dtype=float)
    image_hog = np.zeros(shape=(len(images),box_heigth,box_width), dtype=float)

    j = 0
    for i in images:
        image = []
        box = []
        
        top = labels[j]["y"]
        left = labels[j]["x"]
        width = labels[j]["width"]
        height = labels[j]["height"]
        
        image = io.imread(i)
        image_grey = color_to_grey(image)
        
        max_length = max(width, height)
        ratio = box_width / max_length
        
        
        image_resize = resize(image, (int(len(image[0])*ratio), int(len(image)*ratio)))
    
        
        box = image_grey[top:top+height, left:left+width]
        
        scipy.misc.imsave('outfile.jpg', box)
        print(str(j) + " " + str(top) + " " + str(left) + " " +str(width) + " " +str(height))
        
        box_resize = resize(box, (box_heigth, box_width))
        
        fd_hog[j], image_hog[j] = feature.hog(box_resize, visualise=True)
        print(len(fd_hog))
        
        if(False):
            plt.imshow(image_hog[j])
            plt.show()
            plt.imshow(image)
            plt.show()
            plt.imshow(box_resize)
            plt.show()
        #time.sleep(5)
        j = j+1
    return fd_hog,image_hog
#def get_box()
def color_to_grey(image):
    # on va peut être jouer sur la saturation des couleurs après alors je la met de coté
    return color.rgb2gray(image)

     
def intersect(x1, y1, w1, h1, x2, y2, w2, h2):
    # 1 : fenêtre glissante
    # 2 : label
    #print("x1",x1)
    #print("x2",x2)
    x3 = max(x1, x2)
    y3 = max(y1, y2)

    x4 = min(x1+w1, x2+w2)
    y4 = min(y1+h1, y2+h2)
    
    if ((x3 >= x4) | (y3 >= y4)):
        return 0
    else:
        aire = abs(x4 - x3) * abs(y4 - y3)
        #print("aire", aire)
		
        label_aire = w2 * h2
        ratio = (float(aire)/float(label_aire))*100
        #print("ratio", ratio)
		
        return ratio
    
def sliding_box(image, img, size, jump, bmax, bmin, label_x, label_y, label_width, label_height):
    box_width = size
    box_height = size
    width = max(0, (len(image[0])-box_width))
    height = max(0, (len(image)-box_height))
    
    k=0
    n=0
    p=0
    if ((width != 0) & (height != 0)):
        left = 0
        top = 0
        results = np.zeros(shape=((height//jump)*(width//jump), 3), dtype = int)
        for i in range(0, height//jump):
            for j in range(0, width//jump):
    
                results[k][0] = intersect(left, top, box_width, box_height, label_x, label_y, label_width, label_height)
                results[k][1] = left
                results[k][2] = top
                if ((results[k][0] > bmax) & (p < 3) ):
                    p = p + 1
                    box = image[top:top+box_height, left:left+box_width]
                    scipy.misc.imsave(origin_path+'\\pos\\positive'+str(k)+str(img)+".jpg", box)
                else:
                    if ((results[k][0] < bmin) & (n < 5)):
                        n = n + 1
                        box = image[top:top+box_height, left:left+box_width]
                        scipy.misc.imsave(origin_path+'\\neg\\negative'+str(k)+str(img)+".jpg", box)
                k = k+1
                left = left + jump
            #print("fin ligne")
            left = 0
            top = top + jump
            
        return results

def generate_train_data(path, size, jump, bmax, bmin):
    
    os.chdir(origin_path)
    labels = np.loadtxt("label.txt",dtype={'names': ('name','x', 'y', 'width', 'height'),'formats':('S4','i4','i4','i4','i4')})
    os.chdir(origin_path+path)
    images = glob.glob("*.jpg")
    
    j=0
    for img in images:
        image = []
        box = []
        
        label_y = labels[j]["y"]
        label_x = labels[j]["x"]
        label_width = labels[j]["width"]
        label_height = labels[j]["height"]
    	
        image = io.imread(img)
        image = color.rgb2gray(image)
        
        image_width = len(image[0])
        image_height = len(image)
     
        max_length = max(label_width, label_height)
        ratio = size / max_length
        
        label_height = int(label_height * ratio)
        label_width = int(label_width * ratio)
        label_x = int(label_x * ratio)
        label_y = int(label_y * ratio)
        
        image_resize = resize(image, (int(image_height*ratio), int(image_width*ratio)))
        
        box = image_resize[label_y:label_y+label_height, label_x:label_x+label_width]
        box = resize(box, (size, size))
        scipy.misc.imsave(origin_path+'\\label\\box'+str(img)+'.jpg', box) 
        
        results = sliding_box(image_resize, img, size, jump, bmax, bmin, label_x, label_y, label_width, label_height)
        j= j + 1
        
    return results


#results = generate_train_data("\\train", 32, 2, 90, 15)