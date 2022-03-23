from functools import partial

import pandas as pd
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


def train_step_topo(topo_reg, neuron_space_strategy, model, optimizer,
                    loss_object, train_loss, train_accuracy, inputs, labels, acc, total_occ):
    X = neuron_space_strategy(model, inputs)  # Inputs all batch, with labels X = inputs
    X = tf.Variable(X.array, tf.float64)

    with tf.GradientTape() as tape:
        Dg = RipsModel(X=X, mel=10, dim=0, card=10).call()
        topo_loss = wasserstein_distance(Dg, tf.constant(np.empty([0, 2])), order=1, enable_autodiff=True)

        predictions = _compute_predictions(inputs, model)

        loss = loss_object(labels, predictions)

        loss_topo_reg = (topo_reg * topo_loss) + (1 - topo_reg) * loss

    gradients_topo = tape.gradient(loss_topo_reg, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients_topo, model.trainable_variables))

    train_loss(loss_topo_reg)
    train_accuracy(labels, predictions)
    accuracy_per_class(labels, labels_from_predictions(predictions), acc, total_occ)


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


def test_step(model, images, labels, loss_object, test_loss, test_accuracy, acc, total_occ):
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

    train_loss_topo = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy_topo = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss_topo = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy_topo = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    train_ds = tf.data.Dataset.from_tensor_slices(
        train_dataset).shuffle(10000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices(
        val_dataset).batch(32)

    acc_train_epochs = []
    acc_test_epochs = []
    acc_class_epochs_train = []
    acc_class_epochs_test = []

    acc_train_epochs_topo = []
    acc_test_epochs_topo = []
    acc_class_epochs_train_topo = []
    acc_class_epochs_test_topo = []

    loss_epochs = []
    loss_epochs_topo = []

    for epoch in range(epochs):

        total_occ_train = [0 for x in range(10)]
        acc_train = [np.nan for x in range(10)]
        total_occ_test = [0 for x in range(10)]
        acc_test = [np.nan for x in range(10)]

        total_occ_train_topo = [0 for x in range(10)]
        acc_train_topo = [np.nan for x in range(10)]
        total_occ_test_topo = [0 for x in range(10)]
        acc_test_topo = [np.nan for x in range(10)]

        num = 0

        for inputs, labels in train_ds:
            train_step_topo(topo_reg, neuron_space_strategy, model_topo_reg,
                            optimizer, loss_object, train_loss_topo, train_accuracy_topo,
                            inputs, labels, acc_train_topo, total_occ_train_topo)

            train_step(model_none_topo_reg, optimizer, loss_object,
                       train_loss, train_accuracy, inputs, labels, acc_train, total_occ_train)
            num += 1
            print(num)

        for inputs, labels in test_ds:
            test_step(model_topo_reg, inputs, labels, loss_object, test_loss_topo, test_accuracy_topo,
                      acc_test_topo, total_occ_test_topo)
            test_step(model_none_topo_reg, inputs, labels, loss_object, test_loss, test_accuracy,
                      acc_test, total_occ_test)

        template = 'Epoch {}, Perdida: {}, Exactitud: {}, Perdida de prueba: {}, Exactitud de prueba: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result(),
                              test_loss.result(),
                              test_accuracy.result()))

        template = 'TOPO: Epoch {}, Perdida: {}, Exactitud: {}, Perdida de prueba: {}, Exactitud de prueba: {}'
        print(template.format(epoch + 1,
                              train_loss_topo.result(),
                              train_accuracy_topo.result(),
                              test_loss_topo.result(),
                              test_accuracy_topo.result()))

        acc_train_epochs.append(train_accuracy.result().numpy() * 100)
        acc_test_epochs.append(test_accuracy.result().numpy() * 100)

        acc_train_epochs_topo.append(train_accuracy_topo.result().numpy() * 100)
        acc_test_epochs_topo.append(test_accuracy_topo.result().numpy() * 100)

        loss_epochs.append(train_loss.result())
        loss_epochs_topo.append(train_loss_topo.result())

        acc_train = np.divide(acc_train, total_occ_train)
        acc_test = np.divide(acc_test, total_occ_test)

        acc_train_topo = np.divide(acc_train_topo, total_occ_train_topo)
        acc_test_topo = np.divide(acc_test_topo, total_occ_test_topo)

        print('\n Accuracies per class train: ', acc_train)
        print('\n Accuracies per class test: ', acc_test)
        print('\n Accuracies per class train: ', acc_train_topo)
        print('\n Accuracies per class test: ', acc_test_topo)

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        train_loss_topo.reset_states()
        train_accuracy_topo.reset_states()
        test_loss_topo.reset_states()
        test_accuracy_topo.reset_states()

        acc_class_epochs_train.append(acc_train)
        acc_class_epochs_test.append(acc_test)

        acc_class_epochs_train_topo.append(acc_train_topo)
        acc_class_epochs_test_topo.append(acc_test_topo)

    serialize('AccuracyClassTrain', acc_class_epochs_train)
    serialize('AccuracyClassTest', acc_class_epochs_test)
    serialize('AccuracyTrain', acc_train_epochs)
    serialize('AccuracyTest', acc_test_epochs)

    serialize('AccuracyClassTrainTopo', acc_class_epochs_train_topo)
    serialize('AccuracyClassTestTopo', acc_class_epochs_test_topo)
    serialize('AccuracyTrainTopo', acc_train_epochs_topo)
    serialize('AccuracyTestTopo', acc_test_epochs_topo)

    serialize('LossesEpochs', loss_epochs)
    serialize('LossesEpochsTopo', loss_epochs_topo)


def reduce_dataset(train_images, train_labels, reduction=0.01):
    df = pd.DataFrame(list(zip(train_images, train_labels)), columns=['Image', 'label'])
    val = df.sample(frac=reduction)
    X_train = np.array([i for i in list(val['Image'])])
    y_train = np.array([[i[0]] for i in list(val['label'])])
    return X_train, y_train


if __name__ == '__main__':
    topo_reg = 0.3

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    train_images, train_labels = reduce_dataset(train_images, train_labels)

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
