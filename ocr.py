import cv2 as cv
import numpy as np
import argparse
import time, sys
from PIL import Image
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

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to input image")

args = vars(ap.parse_args())

image = cv.imread("C:\Users\Reshma.Varadaraj\Downloads\MicrosoftTeams-image.png")
(H, W) = image.shape[:2]
print('shape', image.shape)

binThresh = 0.3
polyThresh = 0.5
maxCandidates = 200
unclipRatio = 2


model = cv.dnn_TextDetectionModel_DB('DB_IC15_resnet50.onnx')
#model = cv.dnn_TextDetectionModel_DB('frozen_east_text_detection.pb')

model.setBinaryThreshold(binThresh).setPolygonThreshold(polyThresh).setMaxCandidates(maxCandidates).setUnclipRatio(unclipRatio)

scale = 1.0 / 255.0
mean = 118.2
inputSize = (H, W)

model.setInputParams(scale, (736, 736), (122.67891434, 116.66876762, 104.00698793), swapRB=True, crop=False)

bb, det1 = model.detect(image)

fp = open(args["image"].replace('.jpeg', '.txt'), 'w')

model = cv.dnn_TextRecognitionModel('crnn_cs.onnx')
model.setDecodeType("CTC-greedy")

for k in range(len(bb)):
    flatten_list = [j for sub in bb[k] for j in sub]
    
    box = flatten_list
    print(box[0], box[4], box[1],box[3])
    boxImg = image[box[3]:box[1], box[0]:box[4], :]

    result = recognizer(boxImg, model)
    print(result)
    fp.write('{},{},{},{},{}\n'.format(box[0], box[4], box[1],box[3],result))
    
fp.close()

