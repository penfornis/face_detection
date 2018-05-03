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

min_score = 0.1

train_step = 10
train_limit = 25
negative_nb = 7
#show_mean("train\\pos")

#show_mean("train\\neg")

#print("*** Génération des données ***")

#Générer les images d'entraînement
#fd_hog_pos, fd_hog_neg = data.generate_train_data("\\train", box_width, box_width, train_step, train_limit, negative_nb)

print("*** Evaluation du modèle ***")
###Calculer l'efficacité du modèle
#error, rappel, precision, score = validation_script(fd_hog_pos, fd_hog_neg)
#print("valeurs:", error, precision, score, rappel)

print("*** Préparation des données ***")
#fd_hog, label_hog = label_concat(fd_hog_pos, fd_hog_neg)
##

print("*** Génération du modèle ***")
#clf.fit(fd_hog, label_hog)
##print("*** model OK ***")
print("*** Sauvegarde du modèle ***")
#s = pickle.dump(clf, open (origin_path+"\\save_model.p", "wb"))
clf = pickle.load(open(origin_path+"\\save_model.p", "rb"))

print("*** Génération des résultats ***")
results = detect_face_script(clf, "\\test", box_width, box_height, min_score, 1)
