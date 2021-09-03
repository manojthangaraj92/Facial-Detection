#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import the required libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
from model.package.json import get_json
from model.package.mtcnn import file_read, mtcnn

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
    model = mtcnn(img_path)
    a = x.mtcnn_obj()
    conv = list(a)
    box, probs, landmark = a[0], a[1], a[2]
    get = get_json(box, probs, landmark)
    obj = get.return_result()
    
    return render_template('index.html', prediction_faces='The detected facial features are {}'.format(obj))

if __name__ == "__main__":
    app.run(debug=True)

