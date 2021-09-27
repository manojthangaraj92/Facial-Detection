#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Import the required libraries to build the class
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

#the actual classes for face detection app starts here
class Face_Detector:
    """ 
    A class to detect faces in the picture.

    .....
    Attributes
    ----------
    mtcnn : str
        the face detection library object with it's arguments filled.
    
    Methods
    ----------
    return_frame(boxes,img)
        returns nd array of the image with bounding boxes over the detected faces.

    detect_mtcnn(image, file_path)
        returns the image file name, bounding boxes of faces measurement tensors, the probability score for each face.
    """
    
    def __init__(self,mtcnn):
        """
        Contructs all necessary attributes for the Face_Detector object.

        Parameters
        ------------
            mtcnn : str
                the face detection library object with it's arguments filled.
        """
        self.mtcnn = mtcnn
        
    def return_frame(self,boxes,img):
        """
        This function returns nd array of the image with bounding boxes over the detected faces.

        Parameters
        ------------
        boxes : tensor dtype : float
            The list of tensor equal to the number to detected faces in the picture.
        img : str
            The web address of the image file or the image with .jpg format

        Returns
        ---------
        numpy array of the images
        """
        #to draw a box around faces, loop through the boxes array
        for box in boxes:
            #each array with 4 items, 0 and 2 for the vertical lines. 1 and 3 for the horizontal lines
            x_left = min(box[0],box[2])
            x_right = max(box[0], box[2])
            y_left = min(box[1], box[3])
            y_right = max(box[1],box[3])
            #draw a rectangle around the detected faces using OpenCV rectangle
            self.img = cv2.rectangle(img, (int(x_left), int(y_left)),
                                          (int(x_right), int(y_right)), 
                                          (255, 0, 0), 2)
        return self.img
    
    def detect_mtcnn(self,image,file_path):
        """
        This function detects each faces in the given image.

        Parameters
        -----------
        image : str
            The web address of the image or the image name with the format .jpg.
        file_path : str
            The file path to the local system folder or the database to store the picture.

        Returns
        ----------
        The file name of the stored picture, nd array of bounding boxes and probabilities.
        """
        #try and except methods for the different input types
        try:
            url_response = urllib.request.urlopen(image)
            img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except:
            img = cv2.imread(image,1)
            img = cv2.cvtColor(img, cv2.IMREAD_COLOR)

        # from the facenet pytorch library, to get the bounding boxes and probabilities array, we use method .detect() 
        boxes, probs = self.mtcnn.detect(img)
        
        self.x = self.return_frame(boxes,img)
        x = random.choice(string.ascii_letters)
        jpg_name = f'detected_img_{x}.jpg'
        filename = file_path + jpg_name
        cv2.imwrite(filename, cv2.cvtColor(self.x, cv2.IMREAD_COLOR))
        #return the file name and arrays
        return jpg_name, boxes, probs

class return_json:
    """
    A class to return bounding boxes measurements in JSON format.
    .....
    Attributes
    -----------
    a : array 
        It contains file name, bounding box and probabilities array. 
    
    Methods
    --------
    get_dict(indices)
        Returns dict object with bounding box and probability of detected faces.
    return_result(file_path)
        Returns, written json file name in the local computer or database.
    """
    def __init__(self,a):
        """
        Constructs all necessary atributes for the return_json object

        Parameters
        -----------
        a : arrays
            The file name, bounding boxes array and probability array.
        """
        self.filename = a[0]
        self.boxes = a[1].tolist()
        self.probs = a[2].tolist()
        
    def get_dict(self, indices):
        """
        The function to get a dict object of the bounding boxes and probability arrays.
        
        Parameter
        ----------
        indices : int
            This given indices takes the item from the array and put it the dict object.

        Returns
        ---------
        dict object with bounding boxes measurements and confidence scores.
        """
        json_dict = dict()
        json_dict['box'] = self.boxes[indices]
        json_dict['confidence'] =self.probs[indices]
        return json_dict

    def return_result(self,file_path): 
        """
        This function creates the json file.

        Parameters
        ------------
        file_path : str
            The file path where the created json files to be stored.

        Returns
        ----------
            the json file.
        """
        lists=list()
        for i in range(len(self.boxes)):
            x = self.get_dict(i)
            i+=1
            lists.append(x)

        files = f'/json{random.choice(string.ascii_letters)}.json'
            
        with open(file_path+files, "w") as outfile:
            json.dump(lists, outfile)

        return files
        

