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
    print("début cross validation")

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

def cross_validation_sliding_window(images, labels, N, box_width, box_height):
    ### Paramètres pour la recherche d'images négatives
    #Pas de déplacement de la fenêtre glissante 
    jump = 10
    #Limite d'intersection entre le label et l'image pour qu'elle soit comptée comme négative
    limit = 25
    # Nombre d'images négatives à selectionner par image
    number = 7
    
    print("Début de la validation croisée")
    
    r = np.zeros(N, dtype = float)
    n=0
    predict = np.zeros(len(images)//5, dtype = float)
    
    for i in range(0,N):
        
        print("Génération du modèle sur 4/5 des images")
        
        mask_image = np.zeros(len(images), dtype = bool)
        mask_image[np.arange(i, mask_image.size, N)] = True
        
        part_images = np.array(images)[~mask_image] #4/5 des images
        part_labels = np.array(labels)[~mask_image]
        fd_pos, fd_neg = generate_data(part_images, part_labels, box_width, box_height, jump, limit, number)
        
        fd_hog, label_hog = label_concat(fd_pos, fd_neg)
        
        mask = np.zeros(fd_hog.shape[0], dtype = bool)
        #1/5 des données sont mise à True
        mask[np.arange(i, mask.size, N)] = True
        # on entraine sur 4/5
        clf.fit(fd_hog[~mask,:], label_hog[~mask])
        
        j = 0
        
        print("Test sur 1/5 des images")
        # on teste sur 1/5 des images si on retrouve bien les bons labels
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
            results = sliding_window(clf, image, num, box_width, box_height, 0.5, 1)
            
            window_x = int(results[0][0])
            window_y = int(results[0][1])
            window_width = int(results[0][2])
            window_height = int(results[0][3])
            
            #On calcule l'intersection entre la box trouvée grâce au model et la vraie box
            intersection = intersect(window_x, window_y, window_width, window_height, label_x, label_y, label_width, label_height)
           
            #S'il y a plus de 50% d'intersection
            if intersection > 50:
                predict[j] = 1
                n=n+1
                #On peut enregistrer l'image
                window = image[window_y:window_y+window_height, window_x:window_x+window_width]
                scipy.misc.imsave(origin_path+'\\results\\positive'+str(num)+".jpg", window)
                
            else:
                predict[j] = 0
               
                #on peut enregistrer l'image
                window = image[window_y:window_y+window_height, window_x:window_x+window_width]
                window = color_to_grey(window)
                window = resize(window, (box_height, box_width))
                
                scipy.misc.imsave(origin_path+'\\neg2\\neg2'+str(num)+".jpg", window)
            j = j+1
        r[i] = np.mean(predict != 1) 
       
    error = np.mean(r)*100
    return error, n

def validation_sliding_window_script(path, N, box_width, box_height): 
    labels = get_labels()
    images = get_images(path)
    return cross_validation_sliding_window(images, labels, N, box_width, box_height)  

#error, n = validation_sliding_window_script("\\train", fd_hog_label, fd_hog_neg)