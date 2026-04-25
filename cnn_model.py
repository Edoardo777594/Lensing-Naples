#! -*- coding: utf-8 -*-

import numpy as np
from keras.models import Model
from keras.layers import *
from keras.models import Sequential
from keras import regularizers

#Resnet Conv_Block=2
def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
 
    #x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides,name=conv_name)(x)
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    x = Activation('relu')(x)
    return x
 
        
def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
        x = add([x,shortcut])
        return x
    else:
        x = add([x,inpt])
        return x

def ResNet(inpt):
    #x = ZeroPadding2D((1,1))(inpt)
    x=inpt
    x = Conv2d_BN(x, nb_filter=32, kernel_size=(7, 7), strides=(1, 1))
    x = MaxPooling2D(pool_size=(3, 3),strides=(2, 2),padding='same')(x)
    
    x = Conv_Block(x, nb_filter=32, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True) 
    x = Conv_Block(x, nb_filter=32, kernel_size=(3, 3))
    
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    
    #x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    #x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
   
    #x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    #x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
  
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    
    outpt = Dense(1, activation='sigmoid')(x)
    return outpt
