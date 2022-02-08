import numpy as np
from gudhi.wasserstein import wasserstein_distance

from constraints import EPOCH_ITERATIONS_TO_INCREMENT_ITERATION_BATCHES, SAVING_MODEL_BATCHES_ITERATIONS_PERIOD, \
    POSSIBLE_LEARNING_RATES
from filtrations import RipsModel
from networks.train import _save_model
from point_cloud_diff.diff import diff_point_cloud

import tensorflow as tf
from functools import partial

EPOCHS = 60
TRIAL_EPOCHS = 4


def fibonacci(initial_a=1, initial_b=1):
    a, b = initial_a, initial_b
    while True:
        yield a
        a = b
        a, b = b, a + b


def train_nn_topo(model, train_dataset, dim, distance, topo_penalty,
                  loss_metric_model,
                  accuracy_model,
                  loss_object_model,
                  optimizer=tf.optimizers.Adam,
                  epochs=EPOCHS):
    model = tf.keras.models.clone_model(model)
    loss_object = loss_object_model
    epoch = 0
    predictions = {}

    while epoch < epochs:

        for inputs, label in train_dataset:
            if not predictions[label]:
                predictions[label] = []
            predictions[label].append((_compute_predictions(inputs, model), inputs))

        for label in predictions.keys():
            predictions_point_cloud = [predictions[0] for predictions in predictions[label]]
            inputs = [inputs[1] for inputs in predictions[label]]
            with tf.GradientTape() as tape:
                Dg = RipsModel(X=predictions_point_cloud, mel=10, dim=dim, card=10, distance=distance).call()
                topo_loss = wasserstein_distance(Dg, tf.constant(np.empty([0, 2])), order=1, enable_autodiff=True)
                for _input in inputs:
                    point_loss = loss_object([label], _compute_predictions(_input))
                    loss = topo_penalty*topo_loss + (1 - topo_penalty)*point_loss
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))




@tf.function
def _compute_predictions(inputs, model):
    return model(inputs)


def _train_step(inputs, labels, model, loss_object, optimizer, train_loss=None, train_accuracy=None):
    with tf.GradientTape() as tape:
        predictions = _compute_predictions(inputs, model)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if train_loss:
        train_loss(loss)
    if train_accuracy:
        train_accuracy(labels, predictions)


def _metrics_step(inputs, labels, model, loss_object, train_loss, train_accuracy):
    predictions = _compute_predictions(inputs, model)
    loss = loss_object(labels, predictions)
    train_loss(loss)
    train_accuracy(labels, predictions)


def _test_step(inputs, labels, model, loss_object, test_loss, test_accuracy):
    predictions = _compute_predictions(inputs, model)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


@tf.function
def _calculate_validation_accuracy(inputs, labels, model, loss_object, validation_loss):
    predictions = model(inputs)
    t_loss = loss_object(labels, predictions)
    validation_loss(t_loss)


def _train_trials(model, train_dataset, validation_dataset, epochs, optimizer, loss_metric_model, loss_object_model):
    train_loss = loss_metric_model(name='trial_train_loss')
    validation_loss = loss_metric_model(name='trial_validation_loss')

    loss_object = loss_object_model
    for epoch in range(epochs):
        for inputs, labels in train_dataset:
            _train_step(inputs, labels, model, loss_object, optimizer, train_loss)

        for validation_inputs, validation_labels in validation_dataset:
            _calculate_validation_accuracy(validation_inputs, validation_labels, model, loss_object, validation_loss)
        print("Trial Epoch: ", epoch)
        print("Validation_loss", validation_loss.result())
        train_loss.reset_states()
        if epoch != epochs - 1:
            validation_loss.reset_states()
    validation_result = validation_loss.result().numpy()
    validation_loss.reset_states()
    return float(validation_result)


def _select_best_model_and_optimizer(model, train_dataset, validation_dataset, trial_epochs, optimizer,
                                     loss_metric_model, loss_object_model):
    models_x_optimizers = list(
        map(lambda lr: (tf.keras.models.clone_model(model), optimizer(learning_rate=lr)),
            POSSIBLE_LEARNING_RATES))
    val_losses = list(map(lambda model_x_optimizer:
                          _train_trials(model_x_optimizer[0], train_dataset, validation_dataset, trial_epochs,
                                        model_x_optimizer[1], loss_metric_model, loss_object_model),
                          models_x_optimizers))
    models_x_optimizers_x_val_losses = zip(models_x_optimizers, val_losses)
    return min(models_x_optimizers_x_val_losses, key=lambda model: model[1])[0]



