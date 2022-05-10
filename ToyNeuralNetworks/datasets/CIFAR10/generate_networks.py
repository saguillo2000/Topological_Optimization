import os
import pathlib
import time

from constraints import MAX_NEURONS_FOR_NETWORK, QT_BATCHES, MAX_HIDDEN_LAYERS
from networks.MLP.mlp_generation import generate_networks
from networks.train import _save_model_pre_train
from tensorflow.keras import losses
import tensorflow as tf
from networks.train import train

# Constants for the problem
PROBLEM_INPUT_SIZE = (32, 32, 3)
PROBLEM_OUTPUT_SIZE = 10

ROOT_FOLDER = pathlib.Path().absolute()


def generate_and_save():
    networks = generate_networks(QT_BATCHES,
                                 PROBLEM_INPUT_SIZE,
                                 MAX_HIDDEN_LAYERS,
                                 PROBLEM_OUTPUT_SIZE,
                                 MAX_NEURONS_FOR_NETWORK)

    for network_id, network in enumerate(networks):
        network.compile(optimizer="Adam", loss="mse", metrics=["mae"])
        network.summary()
        _save_model_pre_train(network, network_id, ROOT_FOLDER)


def load_models():
    networks = []

    for file in os.listdir('models'):
        path = os.path.join('models', file)
        temp_network = tf.keras.list_models.load_model(path)
        networks.append(temp_network)


if __name__ == "__main__":
    print("===============================")
    generate_and_save()
    print("All models generated")
    load_models()
