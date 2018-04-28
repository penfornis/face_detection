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
#show_mean("train\\pos")

#show_mean("train\\neg")

print("*** Début ***")

#Générer les images d'entraînement
fd_hog_pos, fd_hog_neg = data.generate_train_data("\\train", 32, 32, 10, 10)

#Calculer l'efficacité du modèle
error, rappel, precision, score = validation_script(fd_hog_pos, fd_hog_neg)

fd_hog, label_hog = label_concat(fd_hog_pos, fd_hog_neg)
clf.fit(fd_hog, label_hog)
#print("*** model OK ***")
s = pickle.dump(clf, open (origin_path+"\\save_model.p", "wb"))

detect_face_script("save_model.p", "\\test", 32, 32)
