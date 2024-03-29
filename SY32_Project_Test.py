# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 16:13:06 2018

@author: Eléonore
"""

from SY32_Project_Tools import *

### Fonctions permettant d'estimer notre modèle

def precision(prediction, truth):
    vp = 0
    fp = 0
    for i in range(0, len(prediction)):
        if (prediction[i] == 1) & (truth[i] == 1):
            vp = vp + 1
        else:
            if (prediction[i] == 1) & (truth[i] == 0):
                fp = fp + 1
    return vp/(vp+fp)
    

def rappel(prediction, truth):
    vp = 0
    fn = 0
    for i in range(0, len(prediction)):
        if (prediction[i] == 1) & (truth[i] == 1):
            vp = vp + 1
        else:
            if (prediction[i] == 0) & (truth[i] == 1):
                fn = fn + 1
    return vp/(vp+fn)

def scoreF1(prediction, truth):
    p = precision(prediction, truth)
    r = rappel(prediction, truth)
    return 2*(p*r)/(p+r)

    
######
#Test du modèle de classifieur
######
    
def cross_validation(x, y, N):
    #x = np.reshape(x, (len(x), 24*24))
    print("Début de la validation croisée")

    r = np.zeros(N, dtype = float)
    predict = np.zeros(len(x), dtype = int)

    for i in range(0,N):
        mask = np.zeros(x.shape[0], dtype = bool)
        mask[np.arange(i, mask.size, N)] = True
        clf.fit(x[~mask,:], y[~mask])
        r[i] = np.mean(clf.predict(x[mask]) != y[mask])
        predict[mask] = clf.predict(x[mask])
    
    #Calcul de l'erreur, de la précision, du rappel et du scoreF1
    error = np.mean(r)*100
    prec = precision(predict, y)
    rap = rappel(predict, y)
    score = scoreF1(predict, y)
    
    return error, prec, rap, score

def validation_script(pos, neg):    
    train, label = label_concat(pos,neg)
    train_s, label_s = shuffle(train,label)
    error, prec, rap, score = cross_validation(train_s, label_s, 5)   
    return error, prec, rap, score

######
#Test de la fenêtre glissante
######

def cross_validation_sliding_window(images, labels, x, y, N, box_width, box_height, min_score, jump):
        
    print("Début de la validation croisée")
    r = np.zeros(N, dtype = float)
    n=0
    predict = np.zeros(len(images)//5, dtype = float)
    
    for i in range(0,N):
        mask = np.zeros(x.shape[0], dtype = bool)
        #1/5 des données sont mise à True
        mask[np.arange(i, mask.size, N)] = True
        # on entraine sur 4/5
        clf.fit(x[~mask,:], y[~mask])
        
        mask_image = np.zeros(len(images), dtype = bool)
        mask_image[np.arange(i, mask_image.size, N)] = True
        j = 0
        
        # on teste sur les images si on retrouve bien les bons labels
        for img in np.array(images)[mask_image]:
            
            num = img.split(".")
            num = int(num[0])
            #On récupère les labels des fichiers
            label_y = labels[num-1]["y"]
            label_x = labels[num-1]["x"]
            label_width = labels[num-1]["width"]
            label_height = labels[num-1]["height"]

            
            # on test sur 1/5
            image = io.imread(img)
            print(num)
            
            #On fait passer la fenêtre glissante
            results = sliding_window(clf, image, num, box_width, box_height, min_score, jump)
            
            if (len(results) != 0):
                window_x = int(results[len(results)-1][0])
                window_y = int(results[len(results)-1][1])
                window_width = int(results[len(results)-1][2])
                window_height = int(results[len(results)-1][3])
                
                #On calcule l'intersection entre la box trouvée grâce au model et la vraie box
                intersection = intersect(window_x, window_y, window_width, window_height, label_x, label_y, label_width, label_height)
               
                #S'il y a plus de 30% d'intersection
                if intersection > 30:
                    predict[j] = 1
                    n=n+1
                    #On peut enregistrer l'image
                    #window = image[window_y:window_y+window_height, window_x:window_x+window_width]
                    #scipy.misc.imsave(origin_path+'\\results\\positive'+str(num)+".jpg", window)
                    
                else:
                    predict[j] = 0
                   
                    #on peut enregistrer l'image
                    #window = image[window_y:window_y+window_height, window_x:window_x+window_width]
                    #scipy.misc.imsave(origin_path+'\\neg2\\neg2'+str(num)+".jpg", window)
            else:
                #Aucun visage n'a été trouvé
                predict[j] = 0
            j = j+1
        r[i] = np.mean(predict != 1) 
       
    error = np.mean(r)*100
    return error, n

def validation_sliding_window_script(path, pos, neg, box_width, box_height, min_score, jump): 
    labels = get_labels()
    images = get_images(path)
    train, label = label_concat(pos,neg)
    return cross_validation_sliding_window(images, labels, train, label, 5, box_width, box_height, min_score, jump)  