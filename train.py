#! -*- coding: utf-8 -*-
import os
import astropy.io.fits as pyfits
from astropy import table
import pandas as pd
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import cnn_model as cmodel
import numpy as np
import augment
import cv2
import csv
from PIL import Image,ImageEnhance

import ast

band=1

X = pd.read_csv('path_to_data',header=None)
Y = pd.read_csv('path_to_data',header=None)
x=[]
y=[]

### Data Augmentation ###
for i in range(len(X)):

    if band==1:
        name = 'C:/Users/Astrolab/Desktop/lens_search_2021/Personal Code/all_data/rband/'+X[0][i]+'_r.fits'
        img= augment.func_data_read(name,band=band)
        x.append(img)
        y.append(Y[0][i])
        aug = augment.flip_col(img)
        x.append(aug)
        y.append(Y[0][i])
        aug1 = augment.rotate_90(img)
        x.append(aug1)
        y.append(Y[0][i])

    else:
        name = 'C:/Users/Astrolab/Desktop/lens_search_2021/Personal Code/all_data/3band/'+X[0][i]+'.png'
        img = augment.func_data_read(name, band=band)
        x.append(img)
        y.append(Y[0][i])
        aug = augment.color_flip(img)
        x.append(aug)
        y.append(Y[0][i])
        aug1 = augment.color_rot(img)
        x.append(aug1)
        y.append(Y[0][i])
#########################

x_train = np.array(x)
y_train = np.array(y)
x_tes = pd.read_csv('C:/Users/Astrolab/Desktop/lens_search_2021/Personal Code/x_val_name.txt',header=None)
y_tes = pd.read_csv('C:/Users/Astrolab/Desktop/lens_search_2021/Personal Code/y_val.txt',header=None)
x_v=[]
y_v=[]
for i in range(len(x_tes)):

    if band==1:
        name ='C:/Users/Astrolab/Desktop/lens_search_2021/Personal Code/all_data/rband/'+x_tes[0][i]+'_r.fits'
        img= augment.func_data_read(name,band=band)
        x_v.append(img)
        y_v.append(y_tes[0][i])

    else:
        name ='C:/Users/Astrolab/Desktop/lens_search_2021/Personal Code/all_data/3band/'+x_tes[0][i]+'.png'
        img = augment.func_data_read(name, band=band)
        x_v.append(img)
        y_v.append(y_tes[0][i])

x_test = np.array(x_v)
y_test = np.array(y_v)




epochs=30
batch_size=64
learning_rate=0.001
steps_train=int(len(x_train)/batch_size)+1
steps_test=int(len(x_test)/batch_size)+1

if band==1:
    inpt = Input(shape=(41,41,1))
    checkpoint = ModelCheckpoint('./model/weights_1band_updated.h5', monitor='val_accuracy',verbose=1, save_best_only=True)
else:
    inpt = Input(shape=(41, 41,3))
    checkpoint = ModelCheckpoint('./model/weights_3band_updated.h5', monitor='val_accuracy',verbose=1, save_best_only=True)

outpt=cmodel.ResNet(inpt)
model = Model(inputs=inpt, outputs=outpt)
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=learning_rate),
              metrics=['accuracy'])

"""if band==1:              
    if os.path.exists('./model/weights_1band.h5'):
        model.load_weights('./model/weights_1band.h5')

        print("checkpoint_loaded")

else:          
    if os.path.exists('./model/weights_3band.h5'):
        model.load_weights('./model/weights_3band.h5')
        print("checkpoint_loaded")"""



print('Training on',len(y_train),'samples','testing on',len(y_test),'samples')

history =model.fit(x=x_train,y=y_train,
                             steps_per_epoch=steps_train, 
                             epochs=epochs,
                             validation_data=(x_test,y_test),
                             validation_steps=steps_test,
                             #use_multiprocessing=True,
                             shuffle=True,
                             callbacks=[checkpoint])

