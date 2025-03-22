import os
import cv2
import sys
import numpy as np
from PIL import Image, ImageEnhance
from keras.preprocessing.image import Iterator
from scipy.ndimage import rotate
from glob import glob




def read_mask_image(path):
    
    img_list =[]
    for file in glob (os.path.join(path,"*.jpg")):
        img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(64,64))
        img = np.asarray(img).astype(np.float32)
        img_list.append(img)

    return img_list     



def read_image(path):

    img_list =[]
    for file in glob (os.path.join(path,"*.jpg")):
        img = cv2.imread(file)
        img = cv2.resize(img,(64,64))
        img = np.asarray(img).astype(np.float32)
        img_list.append(img)

    return img_list 


def get_train_image(image,augmentation =False):

    train =np.array(image) 
        
    if augmentation:
        all_train_img = [train]
        flip_train = train[:,::-1,:]
        all_train_img.append(flip_train)
        
        
        for i in range(0,360,90):
            all_train_img.append(rotate(train,i,axes=(1, 2), reshape=False))
            
            all_train_img.append(rotate(flip_train,i,axes=(1, 2), reshape=False))
            
        train = np.concatenate(all_train_img,axis =0)    
    #            
        return train



def get_mask_image(image,augmentation =False):



    train_mask = np.array(image)/255.0

        assert(np.min(train_mask)==0 and np.max(train_mask)==1)  

        if augmentation:

        all_mask_img =[train_mask]

        flip_mask =train_mask[:,:,::-1]

        all_mask_img.append(flip_mask)

        for i in range(0,360,90):
            all_mask_img.append(rotate(train_mask,i,axes=(1, 2), reshape=False))
            
            all_mask_img.append(rotate(flip_mask,i,axes=(1, 2), reshape=False))  
        train_mask =np.round(np.concatenate(all_mask_img,axis =0))  
            
    return train_mask
