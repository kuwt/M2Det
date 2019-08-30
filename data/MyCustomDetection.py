import os
import pickle
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import uuid

def classMap(inclass):
    outclass = inclass #+ 81
    return outclass

class CustomDetection(data.Dataset):

    def __init__(self, root, preproc=None, target_transform=None, dataset_name='CustomDetection'):
        self.root = root
        self.target_transform = target_transform
        self.preproc = preproc
        
        self.ids = list()      # it is a list of strings
        self.annotations = list() # it is list of numpy.ndarray
        folders = []
        names = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(self.root):
            for folder in d:
                folders.append(os.path.join(r, folder))
                names.append(folder)
        
        print("folders first 5= ")
        for i,folder in enumerate(folders):
            if i >= 5:
                break
            print(folder)

        for folder,name in zip(folders,names):
            imagePath = folder + "/" + name + ".bmp"
            self.ids.append(imagePath)
            annotationPath = folder + "/"+ name + ".txt"
            fo = open(annotationPath, "r")
            lines = fo.readlines()
            groupAnnotation = np.zeros(shape=(len(lines),5),dtype=np.float)
            for i, line in enumerate(lines):
                singleAnnotation = list()
                elements = line.split(',')
                for j,elem in enumerate(elements):
                    groupAnnotation[i][j] = float(elem)
                groupAnnotation[i][4] = classMap(groupAnnotation[i][4])
            self.annotations.append(groupAnnotation);

        print("imageId first 5 = ")
        for i,id in enumerate(self.ids):
            if i >= 5:
                break
            print(id)        

        print("annotations first 5 = ")
        for i,anno in enumerate(self.annotations):
            if i >= 5:
                break
            print(anno)     

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = self.annotations[index]
        img = cv2.imread(img_id, cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        adjustTarget = target
        
        #print("target = ", target )

        if self.target_transform is not None:
            adjustTarget = self.target_transform(adjustTarget)

        if self.preproc is not None:
            img, adjustTarget = self.preproc(img, adjustTarget)

        return img, adjustTarget

    def __len__(self):
        return len(self.ids)
