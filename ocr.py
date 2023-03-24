import cv2 as cv
import numpy as np
import argparse
import time, sys
from PIL import Image
from numpy import asarray
#import chain

def recognizer(img, model):
    
    file = open('alphabet_94.txt', 'r')
    Lines = file.readlines()
    vocabulary = []
    for line in Lines:
	    vocabulary.append(line.strip())
        
    model.setVocabulary(vocabulary)
    scale = 1.0 / 127.5
    model.setInputParams(scale, (100, 32), (127.5, 127.5, 127.5))

    result = model.recognize(img)
    return result 

def imageUploadHandler(img):
    
    img1 = Image.open(img)
    image = asarray(img1)

    print(image)

    binThresh = 0.3
    polyThresh = 0.5
    maxCandidates = 200
    unclipRatio = 2


    model = cv.dnn_TextDetectionModel_DB('DB_IC15_resnet50.onnx')
    #model = cv.dnn_TextDetectionModel_DB('frozen_east_text_detection.pb')

    model.setBinaryThreshold(binThresh).setPolygonThreshold(polyThresh).setMaxCandidates(maxCandidates).setUnclipRatio(unclipRatio)

    scale = 1.0 / 255.0
    mean = 118.2
    model.setInputParams(scale, (736, 736), (122.67891434, 116.66876762, 104.00698793), swapRB=True, crop=False)

    bb, det1 = model.detect(image)


    model = cv.dnn_TextRecognitionModel('crnn_cs.onnx')
    model.setDecodeType("CTC-greedy")
    dict={}
    for k in range(len(bb)):
        flatten_list = [j for sub in bb[k] for j in sub]

        box = flatten_list
        arr = []
        print(box[0], box[4], box[1],box[3])
        arr.append(int(box[0]))
        arr.append(int(box[4]))
        arr.append(int(box[1]))
        arr.append(int(box[3]))
        boxImg = image[box[3]:box[1], box[0]:box[4], :]

        result = recognizer(boxImg, model)
        dict[result]=arr
        print("Result:")
        print(result)

    return dict

from flask import Flask, redirect, url_for, request
app = Flask(__name__)

@app.route('/')
def success():
   return 'Server is running' 

@app.route('/imageUpload',methods = ['POST'])
def image_upload():
    if request.method == 'POST':
        image = request.files['image']
        print(image)
        out = imageUploadHandler(image)
        with open("db.json", "w") as f:
            f.write(str(out))
        return out

@app.route('/getData',methods = ['GET'])
def get_data():
    if request.method == 'GET':
        with open("db.json", "r") as f:
            content = f.read()
        print("Content")
        print(content)
        return content

if __name__ == '__main__':
   app.run(debug = True)