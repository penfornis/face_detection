# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 08:16:36 2018

@author: sy32p009
"""

from SY32_Project_Test import *
from SY32_Project_Tools import *
import SY32_Project_Data as data


clf = svm.LinearSVC()
box_width = 32
box_height = 49

min_score = 0.2

train_step = 10
train_limit = 25
negative_nb = 7
jump = 3

print("*** Génération des données ***")
fd_hog_pos, fd_hog_neg = data.generate_train_data("train", box_width, box_height, train_step, train_limit, negative_nb)

print("*** Evaluation du modèle ***")
error, rappel, precision, score = validation_script(fd_hog_pos, fd_hog_neg)

print("*** Préparation des données ***")
fd_hog, label_hog = label_concat(np.concatenate((fd_hog_pos, fd_sym_pos), axis=0), fd_hog_neg)

print("*** Génération du modèle ***")
clf.fit(fd_hog, label_hog)

print("*** Sauvegarde du modèle ***")
with open ("save_model.p", "wb") as f:
    s = pickle.dump(clf, f)
with open("save_model.p", "rb") as f:
    clf = pickle.load()

print("*** Génération des résultats ***")
results = detect_face_script(clf, "\\test", box_width, box_height, min_score, jump)

print("*** Test de notre programme sur l'ensemble d'entrainement***")
error_script, n = validation_sliding_window_script("train", fd_hog_pos, fd_hog_neg, box_width, box_height)

