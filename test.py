# -*- coding: utf-8 -*-
"""
Created on Sat May  5 14:53:40 2018

@author: Eléonore
"""

origin_path = "C:\\Users\\Eléonore\\Documents\\UTC\\GI04\\SY32\\Projet\\SY32_Reconnaissance_Visages"

clf = svm.LinearSVC()

min_score = 0.2
jump = 3

box_width = 32
box_height = 49

clf = pickle.load(open(origin_path+"\\save_model.p", "rb"))
#
print("*** Génération des résultats ***")
results = detect_face_script(clf, "\\test", box_width, box_height, min_score, jump)

print("*** Test de notre programme sur l'ensemble d'entrainement***")