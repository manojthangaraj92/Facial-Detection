#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import the required libraries
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from model.package.json import get_json
from model.package.mtcnn import file_read, mtcnn, Face_Detector, return_json
from facenet_pytorch import MTCNN
import os
import json


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ''' This will predict the species of the flower'''
    text1 = request.form['image_size']
    text1 = int(text1)
    text2 = request.form['image_margin']
    text2 = int(text2)

    try:
        file = request.files['image']
        img_path = "static/"+file.filename
        file.save(img_path)
          
    except:
        img_path = request.form['url']
    #model = mtcnn(img_path)
    #a = model.mtcnn_obj()
    #conv = list(a)
    #box, probs, landmark = a[0], a[1], a[2]
    #get = get_json(box, probs, landmark)
    #obj = get.return_result()
    x =  MTCNN(image_size=text1, margin=text2, keep_all=True, device='cpu', post_process=False, select_largest=False)
    fcd = Face_Detector(x)
    file_path = 'D:\\Repository\Facial-Detection\static'
    a = fcd.detect_mtcnn(img_path,file_path)
    files, box, probs = a[0], a[1], a[2]
    full_filename = os.path.join('static', files[1:])
    json_s = return_json(a)
    json_result = json_s.return_result(file_path)
    json_files = os.path.join('static', json_result[1:])
    with open(json_files, 'r') as openfile:
  
            # Reading from json file
            your_list = json.load(openfile)

    len_of_faces = len(your_list)

    return render_template('index.html', prediction_text=full_filename, your_list=your_list, faces=f"The predicted number of faces are {len_of_faces}.")
    #return send_file(a, mimetype='image/jpg')
if __name__ == "__main__":
    app.run(debug=True)

