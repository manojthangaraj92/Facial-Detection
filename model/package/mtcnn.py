#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
from PIL import Image
import os
import json
import cv2


class file_read:
    def __init__(self,image):
        self.process(image)
        
    def process(self,image):
        img = cv2.imread(image,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.frame = Image.fromarray(img)
        return self.frame
        
    def display_image(self):
        plt.figure(figsize=(12, 8))
        plt.imshow(self.frame)
        plt.axis('off')

        
class mtcnn(file_read):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network = MTCNN(keep_all=True, device=device, post_process=False, select_largest=False)
    
    def __init__(self,image):
        super().__init__(image)
        
    def process1(self,image):
        super().process()
        
    def display_image(self):
        super().display_image()
        
    def mtcnn_obj(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        network = MTCNN(keep_all=True, device=device, post_process=False, select_largest=False)
        self.boxes, self.probs, self.landmarks = network.detect(self.frame, landmarks=True)
        return self.boxes.tolist(), self.probs.tolist(), self.landmarks.tolist() 

class Face_Detector:
    
    def __init__(self,mtcnn):
        self.mtcnn = mtcnn
        
    def return_frame(self,boxes,img):
        for box in boxes:
            x_left = min(box[0],box[2])
            x_right = max(box[0], box[2])
            y_left = min(box[1], box[3])
            y_right = max(box[1],box[3])
            self.img = cv2.rectangle(img, (int(x_left), int(y_left)),(int(x_right), int(y_right)), 
                                (255, 0, 0), 2)
            #self.frame1 = Image.fromarray(self.img)

        return self.img
    
    def detect_mtcnn(self,image):
        img = cv2.imread(image,1)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, probs = self.mtcnn.detect(img1)
        
        self.x = self.return_frame(boxes,img1)
        frame = Image.fromarray(self.x)
        plt.figure(figsize=(12, 8))
        plt.imshow(frame)
        plt.axis('off')
        

