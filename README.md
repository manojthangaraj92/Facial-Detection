# Facial Detection app using Facenet Pytorch and MTCNN
----------------------------------------------------------------

Detect the faces in the any given picture using this simple app with **Facenet PyTorch library, Flask App**.

## Table of contents

* [Dependencies](#dependencies)
* [Quick Guide](#quick-guide)
* [Detailed Description](#detailed-description)
* [References](#references)

## Dependencies

_To use this app, you may need some dependencies_

Make sure these libraries installed in your computer.

- Pillow.
- facenet_pytorch.
- opencv-python-headless.

 To know more about Facenet PyTorch Library, This [link](https://github.com/timesler/facenet-pytorch) will help.

## Quick Guide

_Upon meeting all the dependencies, you may also need docker app on your local computer for this app to up and running in the local host_.

When docker is installed, using the command prompt or any other of your choice,

- Open cmd.
- Change the current working directory to this repository.
- ```docker compose up``` command will help in up and running of this app.

## Detailed Description

The Face detection app runs based on Facenet PyTorch Library. In addition to that, we will build Flask app in python. This app communicates with the number of following files in the backend. 

It communicates with, 

- index.html
- model

### Model

The model folder consists of the model that detects the face in the given image which is packaged into a module using setup library.

### index.html

The python file, app.py imports the model and gets inputs from the web page, process it and returns the image with the detected faces, and the measurements in the json format in the webpage.
                

## References

* [Facenet-PyTorch](https://github.com/timesler/facenet-pytorch#pretrained-models)
* [MTCNN](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf)