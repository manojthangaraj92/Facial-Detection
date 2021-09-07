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
from matplotlib import pyplot as plt
import random
import string
import urllib


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
    
    def detect_mtcnn(self,image,file_path):
        try:
            url = image
            url_response = urllib.request.urlopen(url)
            img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except:
            img = cv2.imread(image,1)
            img = cv2.cvtColor(img, cv2.IMREAD_COLOR)

        boxes, probs = self.mtcnn.detect(img)
        
        self.x = self.return_frame(boxes,img)
        x = random.choice(string.ascii_letters)
        jpg_name = f'\detected_img_{x}.jpg'
        filename = file_path + jpg_name
        cv2.imwrite(filename, cv2.cvtColor(self.x, cv2.IMREAD_COLOR))
        #file_path = f'static\{filename}'
        return jpg_name, boxes, probs#cv2.imwrite(filename, self.x)

class return_json:
    def __init__(self,a):
        self.filename = a[0]
        self.boxes = a[1].tolist()
        self.probs = a[2].tolist()
        
    def get_dict(self, indices):
        json_dict = dict()
        json_dict['box'] = self.boxes[indices]
        json_dict['confidence'] =self.probs[indices]
        return json_dict
    def return_result(self,file_path): 
        lists=list()
        for i in range(len(self.boxes)):
            x = self.get_dict(i)
            i+=1
            lists.append(x)

        files = f'/json{random.choice(string.ascii_letters)}.json'
            
        with open(file_path+files, "w") as outfile:
            json.dump(lists, outfile)

        return files
        

