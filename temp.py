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
voxels = np.array(voxels)
voxels = voxels.astype(np.float)
# The inputs are 50x50x50 volumes with a single channel, and the
# batch size is 100

class CAE_3D(tf.keras.Model):
    def __init__(self,latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.conv1 = keras.layers.Conv3D(2, 3, activation='relu')
        self.bNorm1 = keras.layers.BatchNormalization()
        self.maxp1 = tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3))
        self.conv2 = keras.layers.Conv3D(4, 5, activation='relu')
        self.bNorm2 = keras.layers.BatchNormalization()
        self.maxp2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))
        self.conv3 = keras.layers.Conv3D(8, 3, activation='relu')
        self.bNorm3 = keras.layers.BatchNormalization()
        self.maxp3 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))
        
        self.flat = keras.layers.Flatten()
        self.latent = keras.layers.Dense(latent_dim)     
        self.fConn1 = keras.layers.Dense(2*2*2*8)
        self.reshape = keras.layers.Reshape(target_shape = (2,2,2,8))
 
        self.upsample1 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))
        self.deconv1 = keras.layers.Conv3DTranspose(4,3,activation='relu')
        self.bNorm4 = keras.layers.BatchNormalization()
        self.upsample2 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))
        self.deconv2 = keras.layers.Conv3DTranspose(8,5,activation='relu')
        self.bNorm5 = keras.layers.BatchNormalization()
        self.upsample3 = tf.keras.layers.UpSampling3D(size=(3, 3, 3))
        self.deconv3 = keras.layers.Conv3DTranspose(1,3,activation='relu')

    
    def call(self, x):
        x = self.conv1(x)
        x = self.bNorm1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.bNorm2(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        x = self.bNorm3(x)
        x = self.maxp3(x)
        
        x = self.flat(x)
        x = self.latent(x)
        x = self.fConn1(x)
        x = self.reshape(x)
        
        x = self.upsample1(x)
        x = self.deconv1(x)
        x = self.bNorm4(x)
        x = self.upsample2(x)
        x = self.deconv2(x)
        x = self.bNorm5(x)
        x = self.upsample3(x)
        x = self.deconv3(x)
        return x
def loss(x, x_bar):
    return keras.losses.mean_squared_error(x, x_bar)
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        reconstruction = model(inputs)
        loss_value = loss(targets, reconstruction)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), reconstruction


latent_dim = 3
model = CAE_3D(latent_dim)
# print(model.summary)
# optimizer = keras.optimizers.Adam(learning_rate=0.001)

# global_step = tf.Variable(0)
# num_epochs = 5
# for epoch in range(num_epochs):
#     print("Epoch: ", epoch)
#     loss_value, grads, reconstruction = grad(model, voxels, voxels)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)
        
#     if global_step.numpy() % 200 == 0:
#         print("Step: {}, Loss: {}".format(global_step.numpy(), loss(voxels, reconstruction).numpy()))
            
# inputShape = [50, 50, 50, 1]
# latent_dim = 3
# model = keras.Sequential([
#     keras.layers.Conv3D(2, 3, activation='relu', input_shape=inputShape),
#     keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3)),
#     keras.layers.Conv3D(4, 5, activation='relu'),
#     keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2)),
#     keras.layers.Conv3D(8, 3, activation='relu'),
#     keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(latent_dim),
#     keras.layers.Dense(2*2*2*8),
#     keras.layers.BatchNormalization(),
#     keras.layers.Reshape(target_shape = (2,2,2,8)),
#     tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
#     keras.layers.Conv3DTranspose(4,3,activation='relu'),
#     keras.layers.BatchNormalization(),
#     tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
#     keras.layers.Conv3DTranspose(8,5,activation='relu'),
#     keras.layers.BatchNormalization(),
#     tf.keras.layers.UpSampling3D(size=(3, 3, 3)),
#     keras.layers.Conv3DTranspose(1,3,activation='relu')
    
# ])
# 
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='MSE')
history = model.fit(voxels, voxels, epochs=100)
print(model.summary())
# print(voxels.shape)

# history = model.fit(voxels, voxels, epochs=1000)

plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# # plt.ylim([0.5, 1])
plt.legend(loc='lower right')