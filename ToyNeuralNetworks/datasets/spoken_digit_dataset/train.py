import pathlib
import time
from functools import partial

from constraints import MAX_NEURONS_FOR_NETWORK, QT_BATCHES, MAX_HIDDEN_LAYERS
from datasets.spoken_digit_dataset.dataset import get_dataset
from networks.MLP.mlp_generation import generate_networks
from tensorflow.keras import losses
import tensorflow as tf
from networks.train import train, fibonacci

BATCHES_INCREMENTATION_STRATEGY = partial(fibonacci, initial_a=1, initial_b=2)

# Constants for the problem
PROBLEM_INPUT_SIZE = (18262,) #Size of the sounds, it can change
PROBLEM_OUTPUT_SIZE = 10

TRAIN_ROOT_FOLDER = pathlib.Path().absolute()


# Functions for training and pick model
def accuracy_model(name):
    return tf.metrics.SparseCategoricalAccuracy(name=name)


def loss_metric_model(name):
    return tf.keras.metrics.Mean(name=name)


loss_object_model = losses.SparseCategoricalCrossentropy(from_logits=True)

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = get_dataset()
    networks = generate_networks(QT_BATCHES,
                                 PROBLEM_INPUT_SIZE,
                                 10,
                                 PROBLEM_OUTPUT_SIZE,
                                 2000)
    print("===============================")
    print("Model loaded. Starting training")
    print("===============================")
    start_training_time = time.perf_counter()
    trained_models = [train(networks[i], train_dataset, val_dataset, i,
                            loss_metric_model, accuracy_model, loss_object_model, TRAIN_ROOT_FOLDER,
                            batches_incrementation_strategy=BATCHES_INCREMENTATION_STRATEGY)
                      for i in range(len(networks))]
    end_training_time = time.perf_counter()
    print(f"Training finished in {end_training_time - start_training_time:0.4f} seconds")
    print("===============================")
    """
    print("Models trained, starting model selection")
    print("===============================")
    start_picking_time = time.perf_counter()
    for i in range(len(networks)):
        pick_and_clean_models(train_dataset, val_dataset, networks[i], i,
                              TRAIN_ROOT_FOLDER, loss_object_model, loss_metric_model)
    end_picking_time = time.perf_counter()
    print(f"Model picking and cleaning finished in {end_picking_time - start_picking_time:0.4f} seconds")
    """