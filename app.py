#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import the required libraries
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from model.package.mtcnn import Face_Detector, return_json
from facenet_pytorch import MTCNN
import os
import json


#Initiate the flask app
app = Flask(__name__)


#route the app to home
@app.route('/')
def index():
    return render_template('index.html')

#route the app to predict method
@app.route('/predict', methods=['POST'])
def predict():
    ''' 
    This function will predict the faces
    '''
    #the image size and margin comes in the format of string and it is converted into integer for the operation
    text1 = request.form['image_size']
    text1 = int(text1)
    text2 = request.form['image_margin']
    text2 = int(text2)

    #try and except method for image and url name
    try:
        file = request.files['image']
        img_path = "static/"+file.filename
        file.save(img_path)
          
    except:
        img_path = request.form['url']

    #Initiate the MTCNN object from Facenet PyTorch Library which detects faces in the given image
    x =  MTCNN(image_size=text1, margin=text2, keep_all=True, device='cpu', post_process=False, select_largest=False)
    # Send the MTCNN object through the Face_Detector Class
    fcd = Face_Detector(x)
    # Set the file path for the files to be saved
    file_path = 'static/'
    # Use the face detector class detect_mtcnn method to detect faces
    a = fcd.detect_mtcnn(img_path,file_path)
    #from above returned tuple, assign the filename, bounding boxes and probabilities accordingly 
    files, box, probs = a[0], a[1], a[2]
    #get the filename of the written image
    full_filename = os.path.join('static', files)
    #use the return_json class to get the json file by using return_result method
    json_s = return_json(a)
    json_result = json_s.return_result(file_path)
    json_files = os.path.join('static', json_result[1:])
    #open the json file
    with open(json_files, 'r') as openfile:
        # Reading from json file
        your_list = json.load(openfile)

    len_of_faces = len(your_list)

    #return the results in the html page
    return render_template('index.html', prediction_text=full_filename, your_list=your_list, faces=f"The predicted number of faces are {len_of_faces}.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

