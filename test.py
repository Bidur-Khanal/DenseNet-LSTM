import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import os
from keras.models import *
from keras import backend as keras
import pandas as pd
from keras import callbacks
from LSTM import Densenet_LSTM
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image



if __name__ == "__main__":

    ### prediction..
    
    image_path = 'try.jpg'
    im = Image.open(image_path)
    im = np.array(im)
    im = np.delete(im, [1, 2], axis=2)
    im = np.array(im) / 255.0
    im = np.expand_dims(im, axis=0)
    print(im.shape)
    model = Densenet_LSTM()
    model.load_weights("outputs/model-230.h5")
    
    
   
    lmarks = model.predict(im)
    print (lmarks)
    lmarks= lmarks[0]
    lmarks[0:8:2] = lmarks[0:8:2] * im.shape[2]
    lmarks[1:8:2] = lmarks[1:8:2] * im.shape[1]
    print (lmarks)
    
   
    #print(lmarks)
    im = im[0] * 255
    im = np.squeeze(im, axis=(2,))
    print(im.shape)
    for m in range(0, 8,2):
        cv2.circle(im, (int(lmarks[m]), int(lmarks[m + 1])), 5, (255, 255, 255), -1)

    cv2.imwrite('landmarks.png',im)
