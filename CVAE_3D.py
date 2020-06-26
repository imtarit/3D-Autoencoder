# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:11:49 2020

@author: fayeem
"""
import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
import time

voxles = scipy.io.loadmat('voxels.mat')
voxels = voxles["voxels"]

voxels = voxels[..., np.newaxis]
voxels = np.array(voxels)
voxels = voxels.astype(np.float)

class CVAE_3D(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim):
    super().__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            keras.layers.Conv3D(2, 3, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool3D(pool_size=(3, 3, 3)),
            keras.layers.Conv3D(4, 5, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool3D(pool_size=(2, 2, 2)),
            keras.layers.Conv3D(8, 3, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool3D(pool_size=(2, 2, 2)),
        
            keras.layers.Flatten(),
            keras.layers.Dense(latent_dim)
        ]
    )

    self.decoder = tf.keras.Sequential(
        [  
            keras.layers.Dense(2*2*2*8),
            keras.layers.Reshape(target_shape = (2,2,2,8)),
 
            keras.layers.UpSampling3D(size=(2, 2, 2)),
            keras.layers.Conv3DTranspose(4,3,activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.UpSampling3D(size=(2, 2, 2)),
            keras.layers.Conv3DTranspose(8,5,activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.UpSampling3D(size=(3, 3, 3)),
            keras.layers.Conv3DTranspose(1,3,activation='relu')
        ]
    )

  @tf.function
  def encode(self, x):
    logits = self.encoder(x)
    return logits

  def decode(self, z):
    logits = self.decoder(z)
    return logits

optimizer = tf.keras.optimizers.Adam(1e-4)

def compute_loss(model, x):
  code = model.encode(x)
  output = model.decode(code)
  loss = keras.losses.MSE(x,output)
  return loss

@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

epochs = 2
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 3

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
model = CVAE_3D(latent_dim)

def generate_and_plot_code(model, test_sample):
  code = model.encode(test_sample)
  return code


for epoch in range(1, epochs + 1):
  start_time = time.time()
  train_step(model, voxels, optimizer)
  end_time = time.time()
  loss = keras.metrics.Mean()
  epochLoss = loss(compute_loss(model, voxels))
  print('Epoch: {}, time elapse for current epoch: {}, loss: {}'.format(epoch, end_time - start_time, epochLoss))

code = generate_and_plot_code(model, voxels)
print(code)