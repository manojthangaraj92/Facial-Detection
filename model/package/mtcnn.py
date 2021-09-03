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
        

