# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 08:16:36 2018

@author: sy32p009
"""

from SY32_Project_Tools import *
import SY32_Project_Data as data

origin_path = "C:\\Users\\Eléonore\\Documents\\UTC\\GI04\\SY32\\Projet\\SY32_Reconnaissance_Visages"
clf = svm.LinearSVC()
#show_mean("train\\pos")

#show_mean("train\\neg")

print("*** Début ***")
 
fd_hog_label = compute_hog("\\label") 
#                                                           
#print("*** fd_hog_pos OK ***")
fd_hog_neg = compute_hog("\\neg") 
#
#print("*** fd_hog_neg OK ***")
#
#error = validation_script(fd_hog_pos, fd_hog_neg)

#
#
fd_hog, label_hog = label_concat(fd_hog_label, fd_hog_neg)
#
#print("*** Compute model ***")
clf.fit(fd_hog, label_hog)
#print("*** model OK ***")
s = pickle.dump(clf, open ("save_model_label.p", "wb"))
#   
##pos_test = compute_hog("../test")
##neg_test = compute_hog("../test")
#pos_test = compute_hog("test\\pos")
#neg_test = compute_hog("test\\neg")
#
#fd_hog_test, label_hog_test = label_concat(pos_test, neg_test)
#
#nb_error = np.mean(clf.predict(fd_hog_test) != label_hog_test)
#
