# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 08:16:36 2018

@author: sy32p009
"""

from SY32_Project_Tools import *
import SY32_Prject_Data as data
 
#show_mean("train\\pos")

#show_mean("train\\neg")


#pos = read_img_float("train\\pos")
#neg = read_img_float("train\\neg")

fd_hog_pos = compute_hog("train\\pos")
fd_hog_neg = compute_hog("train\\neg")

fd_hog, label_hog = label_concat(fd_hog_pos, fd_hog_neg)

model_hog = clf.fit(fd_hog, label_hog)

#s = pickle.dumps(clf)
   
pos_test = compute_hog("test\\pos")
neg_test = compute_hog("test\\neg")

fd_hog_test, label_hog_test = label_concat(pos_test, neg_test)

nb_error = np.mean(clf.predict(fd_hog_test) != label_hog_test)
