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


inputShape = [50, 50, 50, 1]
latent_dim = 3
model = keras.Sequential([
    keras.layers.Conv3D(2, 3, activation='relu', input_shape=inputShape),
    keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3)),
    keras.layers.Conv3D(4, 5, activation='relu'),
    keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2)),
    keras.layers.Conv3D(8, 3, activation='relu'),
    keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(latent_dim),
    keras.layers.Dense(2*2*2*8),
    keras.layers.BatchNormalization(),
    keras.layers.Reshape(target_shape = (2,2,2,8)),
    tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
    keras.layers.Conv3DTranspose(4,3,activation='relu'),
    keras.layers.BatchNormalization(),
    tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
    keras.layers.Conv3DTranspose(8,5,activation='relu'),
    keras.layers.BatchNormalization(),
    tf.keras.layers.UpSampling3D(size=(3, 3, 3)),
    keras.layers.Conv3DTranspose(1,3,activation='relu')
    
])
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='MSE')
print(model.summary())
# print(voxels.shape)

history = model.fit(voxels, voxels, epochs=1000)

plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# # plt.ylim([0.5, 1])
plt.legend(loc='lower right')