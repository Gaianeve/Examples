import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D

## Modello del q_network
def q_network(screen_shape, learning_rate):
    #input
    start_state = Input(shape = screen_shape)

    #initialize weights
    w_start = tf.keras.initializers.VarianceScaling()

    #layers
    x = Conv2D(32, kernel_size = (8,8), strides = 4, padding = 'SAME',kernel_initializer =w_start, name = 'conv_1' )(start_state) #first layer
    x = Conv2D(64, kernel_size = (4,4), strides = 2, padding = 'SAME',kernel_initializer =w_start, name = 'conv_2' )(x) #second layer
    x = Conv2D(64, kernel_size = (3,3), strides = 1, padding = 'SAME',kernel_initializer =w_start, name = 'conv_3' )(x) #thrid layer
    flatten = Flatten(name='flatten')(x) #flatten output
    fully_connected = Dense(128, kernel_initializer =w_start, name = 'fc')(flatten)  #fully connected layer
    out_layer = Dense(18,kernel_initializer =w_start, name = 'output')(fully_connected) #output

    model = Model(start_state, out_layer)

    #summarize model
    model.summary()

    #compile model with handmade loss function
    model.compile(optimizer= Adam(lr = learning_rate), loss = "mse")

    return model



##main
input_shape = (88,80,1)

model = q_network(input_shape,0.1)

