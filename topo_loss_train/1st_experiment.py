import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

from ToyNeuralNetworks.datasets.CIFAR10.dataset import get_dataset

if __name__ == '__main__':

    train_dataset, val_dataset, test_dataset = get_dataset()

    model = tf.keras.Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), input_shape=(224, 224, 3), activation='relu'),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2), padding='same'),

        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2), padding='same'),

        Flatten(),
        Dense(units=256, activation='relu'),
        Dense(units=512, activation='relu'),
        Dense(units=128, activation='relu'),
        Dense(units=10, activation='softmax')  # CIFAR 10
    ])





    print('Finished')
