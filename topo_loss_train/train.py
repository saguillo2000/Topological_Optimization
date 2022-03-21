from functools import partial

import tensorflow as tf
import numpy as np

from gudhi.wasserstein import wasserstein_distance
from keras.models import clone_model
from tensorflow.keras import datasets

from filtrations import RipsModel
from model import NeuronSpace
from model.NeuronClusteringStrategies import AverageImportanceStrategy
from tensorflow.keras import losses

import pickle

import warnings

warnings.filterwarnings("ignore")


def accuracy_per_class(true, pred, acc, total_occ):
    # We start with nan as in this label a class may not appear
    true = true.numpy().flatten()
    pred = pred.numpy().flatten()

    for truth, prediction in zip(true, pred):

        if np.isnan(acc[truth]):
            acc[truth] = 0

        if truth == prediction:
            acc[truth] += 1

        total_occ[truth] += 1


def accuracy_model(name):
    return tf.metrics.SparseCategoricalAccuracy(name=name)


def labels_from_predictions(_predictions):
    return tf.expand_dims(tf.math.argmax(_predictions, 1), 1)


def train_step_topo(topo_reg, neuron_space_strategy, model, optimizer, loss_object, inputs, labels):
    X = neuron_space_strategy(model, inputs)  # Inputs all batch, with labels X = inputs
    X = tf.Variable(X.array, tf.float64)

    with tf.GradientTape() as tape:
        Dg = RipsModel(X=X, mel=10, dim=0, card=10).call()
        topo_loss = wasserstein_distance(Dg, tf.constant(np.empty([0, 2])), order=1, enable_autodiff=True)

        predictions_topo_reg = _compute_predictions(inputs, model)

        single_loss_topo_reg = loss_object(labels, predictions_topo_reg)

        loss_topo_reg = (topo_reg * topo_loss) + (1 - topo_reg) * single_loss_topo_reg

    gradients_topo = tape.gradient(loss_topo_reg, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients_topo, model.trainable_variables))


@tf.function
def _compute_predictions(inputs, model):
    return model(inputs)


def train_step(model, optimizer, loss_object, train_loss, train_accuracy, inputs, labels, acc, total_occ):
    with tf.GradientTape() as tape:
        predictions = _compute_predictions(inputs, model)

        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
    accuracy_per_class(labels, labels_from_predictions(predictions), acc, total_occ)


def test_step(images, labels, loss_object, test_loss, test_accuracy, acc, total_occ):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    accuracy_per_class(labels, labels_from_predictions(predictions), acc, total_occ)


def serialize(fileName, content):
    outputFile = open('{}.pkl'.format(fileName), 'wb')
    pickle.dump(content, outputFile)
    outputFile.close()


def train_experiment(epochs, topo_reg, loss_object, model, optimizer,
                     train_dataset, val_dataset, neuron_space_strategy):
    model_topo_reg = clone_model(model)
    model_none_topo_reg = clone_model(model)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    train_ds = tf.data.Dataset.from_tensor_slices(
        train_dataset).shuffle(10000).batch(32, drop_remainder=True)

    test_ds = tf.data.Dataset.from_tensor_slices(val_dataset).batch(2, drop_remainder=True)

    acc_class_train = []
    acc_class_test = []

    for epoch in range(epochs):

        total_occ_train = [0 for x in range(10)]
        acc_train = [np.nan for x in range(10)]

        total_occ_test = [0 for x in range(10)]
        acc_test = [np.nan for x in range(10)]

        for inputs, labels in train_ds:
            # train_step_topo(topo_reg, neuron_space_strategy, model_topo_reg, optimizer, loss_object, inputs, labels)
            train_step(model_none_topo_reg, optimizer, loss_object,
                       train_loss, train_accuracy, inputs, labels, acc_train, total_occ_train)

            predictions = _compute_predictions(inputs, model_none_topo_reg)
            predictions_topo_reg = _compute_predictions(inputs, model_topo_reg)
            # print('Labels: ', labels)
            # print('Predictions None reg: ', predictions)
            # print('Predictions Reg: ', predictions_topo_reg)

        for inputs, labels in test_ds:
            test_step(inputs, labels, loss_object, test_loss, test_accuracy,
                      acc_test, total_occ_test)

        template = 'Epoch {}, Perdida: {}, Exactitud: {}, Perdida de prueba: {}, Exactitud de prueba: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result(),
                              test_loss.result(),
                              test_accuracy.result()))
        print('\n Accuracies per class train: ', np.divide(acc_train, total_occ_train))
        print('\n Accuracies per class test: ', np.divide(acc_test, total_occ_test))

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        acc_class_train.append(np.divide(acc_train, total_occ_train))
        acc_class_test.append(np.divide(acc_test, total_occ_test))

    serialize('AccuracyClassTrain', acc_class_train)
    serialize('AccuracyClassTest', acc_class_test)


if __name__ == '__main__':
    topo_reg = 0.3

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    train_dataset = (train_images, train_labels)
    val_dataset = (test_images, test_labels)

    accuracy_model = accuracy_model(name='accuracy')

    loss_object = losses.SparseCategoricalCrossentropy()

    # generate_networks(1, (32, 32, 3), 8, 10, 4000)[0]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=[32, 32, 3]),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam()

    clustering_strategy = partial(AverageImportanceStrategy.average_importance_clustering, number_of_neurons=3000)
    neuron_space_strategy = partial(NeuronSpace.get_neuron_activation_space_with_clustering,
                                    neuron_clustering_strategy=clustering_strategy)

    print('MODEL ARCHITECTURE FOR TOPO REG AND NONE TOPO REG: ')
    print(model.summary())

    train_experiment(20, topo_reg, loss_object, model, optimizer,
                     train_dataset, val_dataset, neuron_space_strategy)
