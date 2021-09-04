#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import the required libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
from model.package.json import get_json
from model.package.mtcnn import file_read, mtcnn, Face_Detector
from facenet_pytorch import MTCNN

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ''' This will predict the species of the flower'''
    file = request.files['image']
    img_path = "static/"+file.filename
    file.save(img_path)
    #model = mtcnn(img_path)
    #a = model.mtcnn_obj()
    #conv = list(a)
    #box, probs, landmark = a[0], a[1], a[2]
    #get = get_json(box, probs, landmark)
    #obj = get.return_result()
    x =  MTCNN(keep_all=True, device='cpu', post_process=False, select_largest=False)
    fcd = Face_Detector(mtcnn)
    a = fcd.detect_mtcnn('123.jpg')
    return render_template('index.html', prediction_text='The detected facial features are {}'.format(a))

if __name__ == "__main__":
    app.run(debug=True)

