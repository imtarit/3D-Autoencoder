# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:19:46 2020

@author: Fayeem
"""

import scipy.io
import numpy as np
import tensorflow as tf

voxles = scipy.io.loadmat('voxels.mat')
voxels = voxles["voxels"]

voxels = voxels[..., np.newaxis]

print(voxels.shape)

# The inputs are 50x50x50 volumes with a single channel, and the
# batch size is 100

# def build_model():
#   model = keras.Sequential([
#     layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(1)
#   ])
  

input_shape =(100, 50, 50, 50, 1)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv3D(
2, 3, activation='relu', input_shape=input_shape)(x)
print(y.shape)

# tf.keras.layers.Conv3DTranspose(
#     filters, kernel_size, strides=(1, 1, 1), padding='valid', output_padding=None,
#     data_format=None, activation=None, use_bias=True,
#     kernel_initializer='glorot_uniform', bias_initializer='zeros',
#     kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
#     kernel_constraint=None, bias_constraint=None, **kwargs
# )
tf.keras.layers.Conv3DTranspose(
    filters, kernel_size, strides=(1, 1, 1), padding='valid', output_padding=None,
    data_format=None, activation=None, use_bias=True,
    kernel_initializer='glorot_uniform', bias_initializer='zeros',
    kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None, **kwargs
)