# -*- coding: utf-8 -*-
"""
Created on Sat May  5 14:53:30 2018

@author: Eléonore
"""

from SY32_Project_Test import *
from SY32_Project_Tools import *
import SY32_Project_Data as data

origin_path = "C:\\Users\\Eléonore\\Documents\\UTC\\GI04\\SY32\\Projet\\SY32_Reconnaissance_Visages"


clf = svm.LinearSVC()

train_step = 10
train_limit = 25
negative_nb = 7

min_score = 0.2
jump = 3

#show_mean("\\label")

print("*** Génération des données ***")
#fd_hog_pos, fd_hog_neg = data.generate_train_data("\\train", box_width, box_height, train_step, train_limit, negative_nb)

print("*** Evaluation du modèle ***")
#error, rappel, precision, score = validation_script(fd_hog_pos, fd_hog_neg)

#print("*** Evaluation du détecteur ***")
#error_script, n = validation_sliding_window_script("\\train", fd_hog_pos, fd_hog_neg, box_width, box_height, min_score, jump)

print("*** Préparation des données ***")
fd_hog, label_hog = label_concat(np.concatenate((fd_hog_pos, fd_sym_pos), axis=0), fd_hog_neg)

print("*** Génération du modèle ***")
clf.fit(fd_hog, label_hog)

print("*** Sauvegarde du modèle ***")
s = pickle.dump(clf, open (origin_path+"\\save_model.p", "wb"))



