#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import the required libraries
import numpy as np
#from flask import Flask, request, jsonify, render_template, send_file
#from model.package.json import get_json
from model.package.mtcnn import Face_Detector, return_json
from facenet_pytorch import MTCNN
import os
import json
import streamlit as st
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image, ImageDraw, ImageFont
import cv2

st.header("Facial Detection with MTCNN")
url = st.text_input('Enter some url:')



def detect(url):
    ''' This will predict the species of the faces'''
    #image = cv2.imread(url)
    x =  MTCNN(image_size=160, margin=20, keep_all=True, device='cpu', post_process=False, select_largest=False)
    fcd = Face_Detector(x)
    a = fcd.detect_mtcnn(url)
    
    
    json_s = return_json(a)
    json_result = json_s.return_result(file_path)
    json_files = os.path.join('static', json_result[1:])
    with open(json_files, 'r') as openfile:
  
            # Reading from json file
            your_list = json.load(openfile)

    len_of_faces = len(your_list)

if url is not None and len(url)>1:

    detect(url)
