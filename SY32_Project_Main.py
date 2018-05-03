# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 08:16:36 2018

@author: sy32p009
"""

from SY32_Project_Test import *
from SY32_Project_Tools import *
import SY32_Project_Data as data

origin_path = "C:\\Users\\Eléonore\\Documents\\UTC\\GI04\\SY32\\Projet\\SY32_Reconnaissance_Visages"
clf = svm.LinearSVC()
box_width = 32
box_height = 32

r_test_init = 0.6
r_test_step = 0.4
min_score = 0.2

r_train_init = 0
r_train_step = 1

train_step = 25
#show_mean("train\\pos")

#show_mean("train\\neg")

print("*** Début ***")

#Générer les images d'entraînement
#fd_hog_pos, fd_hog_neg = data.generate_train_data("\\train", 32, 32, 10, 25)
##
###Calculer l'efficacité du modèle
#error, rappel, precision, score = validation_script(fd_hog_pos, fd_hog_neg)
#print("valeurs:", error, precision, score, rappel)
##
fd_hog, label_hog = label_concat(fd_hog_pos, fd_hog_neg)
##
clf.fit(fd_hog, label_hog)
##print("*** model OK ***")
s = pickle.dump(clf, open (origin_path+"\\save_model.p", "wb"))
clf = pickle.load(open(origin_path+"\\save_model.p", "rb"))
results = detect_face_script(clf, "\\test", 32, 32, 1)
#fd_neg3, error_script, n = validation_sliding_window_script("\\train", fd_hog_pos, fd_hog_neg, 32, 32) 


#results = detect_face_script(clf, "\\test", 32, 32, 2)
### Surapprentissage
#fd_hog_neg_bis = np.concatenate((fd_hog_neg, fd_neg3), axis=0)


#error_bis, rappel_bis, precision_bis, score_bis = validation_script(fd_hog_pos, fd_hog_neg_bis)
#
#fd_hog, label_hog = label_concat(fd_hog_pos, fd_hog_neg_bis)
#
#clf.fit(fd_hog, label_hog)
#print("*** model OK ***")
#s = pickle.dump(clf, open (origin_path+"\\save_model.p", "wb"))
#clf = pickle.load(open(origin_path+"\\save_model.p", "rb"))
#
