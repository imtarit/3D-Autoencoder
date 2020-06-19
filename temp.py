# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:19:46 2020

@author: Fayeem
"""

import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 

voxles = scipy.io.loadmat('voxels.mat')
voxels = voxles["voxels"]

voxels = voxels[..., np.newaxis]

# The inputs are 50x50x50 volumes with a single channel, and the
# batch size is 100

# def build_model():
#   model = keras.Sequential([
#     layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(1)
#   ])
  

# input_shape =(100, 50, 50, 50, 1)
# x = tf.random.normal(input_shape)
# print(type(x))

# tf.keras.layers.Conv3D(
#     filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None,
#     dilation_rate=(1, 1, 1), activation=None, use_bias=True,
#     kernel_initializer='glorot_uniform', bias_initializer='zeros',
#     kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
#     kernel_constraint=None, bias_constraint=None, **kwargs
# )

# def autoencoder_3D(inputShape = [100,50,50,1]):
#   model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv3D(2, 3, activation='relu', input_shape=inputShape),
#     tf.keras.layers.Conv3DTranspose(1,3,activation='relu')
#   ])
#   model.compile(optimizer='adam',
#               loss=tf.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#   return model

# layer_1 = tf.keras.layers.Conv3D(2, 3, activation='relu', input_shape=voxels.shape)
# y = layer_1(voxels.astype(np.float32))
# print(y.shape)

# tf.keras.layers.Conv3DTranspose(
#     filters, kernel_size, strides=(1, 1, 1), padding='valid', output_padding=None,
#     data_format=None, activation=None, use_bias=True,
#     kernel_initializer='glorot_uniform', bias_initializer='zeros',
#     kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
#     kernel_constraint=None, bias_constraint=None, **kwargs
# # )
# layer_2 = tf.keras.layers.Conv3DTranspose(1,3,activation='relu')
# z = layer_2(y)
# print(z.shape)

# model = autoencoder_3D([60,20,20,1])
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses())

inputShape = [50, 50, 50, 1]
latent_dim = 3
model = keras.Sequential([
    keras.layers.Conv3D(2, 3, activation='relu', input_shape=inputShape),
    keras.layers.Conv3D(4, 3, activation='relu'),
    keras.layers.Conv3D(8, 3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(latent_dim),
    keras.layers.Dense(44*44*44*8),
    keras.layers.Reshape(target_shape = (44,44,44,8)),
    keras.layers.Conv3DTranspose(4,3,activation='relu'),
    keras.layers.Conv3DTranspose(2,3,activation='relu'),
    keras.layers.Conv3DTranspose(1,3,activation='relu')
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='MSE')
print(model.summary())
print(voxels.shape)

history = model.fit(voxels, voxels, epochs=10)

plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
plt.legend(loc='lower right')