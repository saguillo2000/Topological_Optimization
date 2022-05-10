from functools import partial

import tensorflow as tf
import numpy as np
import os

from constraints import EPOCHS, TRIAL_EPOCHS, MODEL_FILENAME_PATH, MODEL_FILENAME_STRUCTURE, POSSIBLE_LEARNING_RATES, \
    EPOCH_ITERATIONS_TO_INCREMENT_ITERATION_BATCHES, SAVING_MODEL_BATCHES_ITERATIONS_PERIOD, MODEL_FILENAME_PATH_MODELS, \
    MODEL_FILENAME_PATH_GEN


def fibonacci(initial_a=1, initial_b=1):
    a, b = initial_a, initial_b
    while True:
        yield a
        a = b
        a, b = b, a + b


"""
Note that accuracy_model and loss_metric_model will be callables that receive a name and expect to receive a 
keras 'metrics'. See the examples in any train file of any dataset. 

The loss_object_model is a keras 'losses'.
"""


def train(model, train_dataset, validation_dataset, network_number,
          loss_metric_model,
          accuracy_model,
          loss_object_model,
          train_root_folder,
          optimizer=tf.optimizers.Adam,
          epochs=EPOCHS,
          trial_epochs=TRIAL_EPOCHS,
          batches_incrementation_strategy=partial(fibonacci, initial_a=34, initial_b=55)):
    train_dataset_list = list(train_dataset)

    def batches_generator(number_of_batches):
        current_batch = 0
        while True:
            epoch_changed = False
            train_batches = []
            until_idx = (current_batch + number_of_batches) % len(train_dataset_list)
            if until_idx <= current_batch:
                train_batches.extend(train_dataset_list[current_batch:])
                train_batches.extend(train_dataset_list[:until_idx])
                epoch_changed = True
            elif until_idx > current_batch:
                train_batches.extend(train_dataset_list[current_batch:until_idx])
            current_batch = until_idx
            yield train_batches, epoch_changed

    _, optimizer = _select_best_model_and_optimizer(model, train_dataset, validation_dataset, trial_epochs,
                                                    optimizer, loss_metric_model, loss_object_model)
    model = tf.keras.list_models.clone_model(model)
    batches_incrementation_strategy = batches_incrementation_strategy()
    train_loss = loss_metric_model(name='train_loss')
    train_accuracy = accuracy_model(name='train_accuracy')

    validation_loss = loss_metric_model(name='validation_loss')
    validation_accuracy = accuracy_model(name='validation_accuracy')

    loss_object = loss_object_model
    epoch = 0
    number_of_batches_iterations = 0
    validation_accuracies = []
    train_accuracies = []
    train_losses = []
    validation_losses = []

    number_of_batches_changed = False
    while epoch < epochs:
        if epoch % EPOCH_ITERATIONS_TO_INCREMENT_ITERATION_BATCHES == 0 and (not number_of_batches_changed):
            number_of_batches = next(batches_incrementation_strategy)
            number_of_batches = min(len(train_dataset), number_of_batches)
            assert number_of_batches <= len(train_dataset)
            train_batches = batches_generator(number_of_batches)
            number_of_batches_changed = True
        batches, changed_epoch = next(train_batches)
        for inputs, labels in batches:
            _train_step(inputs, labels, model, loss_object, optimizer)

        for inputs_train_metrics, labels_train_metrics in train_dataset:
            _metrics_step(inputs_train_metrics, labels_train_metrics, model, loss_object, train_loss, train_accuracy)

        for validation_inputs, validation_labels in validation_dataset:
            _test_step(validation_inputs, validation_labels, model, loss_object, validation_loss, validation_accuracy)

        # Saving data about iterations
        train_losses.append(train_loss.result().numpy())
        train_accuracies.append(train_accuracy.result().numpy())
        validation_losses.append(validation_loss.result().numpy())
        validation_accuracies.append(validation_accuracy.result().numpy())

        # Printing data about iterations
        template = 'Epoch {}, Iteration {} Train loss: {}, Accuracy: {}, Test loss: {}, Validation accuracy: {}'
        print(template.format(epoch + 1,
                              number_of_batches_iterations + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              validation_loss.result(),
                              validation_accuracy.result() * 100))
        # Changing iteration
        number_of_batches_iterations += 1

        # Changing the epoch if necessary
        if changed_epoch:
            number_of_batches_changed = False
            epoch += 1

        # Restart the metrics for the next iteration
        train_loss.reset_states()
        train_accuracy.reset_states()
        validation_loss.reset_states()
        validation_accuracy.reset_states()

        # Saving the model if necessary
        if number_of_batches_iterations % SAVING_MODEL_BATCHES_ITERATIONS_PERIOD == 0:
            _save_model(model, network_number, number_of_batches_iterations, train_root_folder)
            print("Saved model")

    # Save metrics into disc
    validation_accuracies = np.array(validation_accuracies)
    train_accuracies = np.array(train_accuracies)
    train_losses = np.array(train_losses)
    validation_losses = np.array(validation_losses)

    _save_np_array(validation_accuracies, "validation_accuracies", network_number, train_root_folder)
    _save_np_array(train_accuracies, "train_accuracies", network_number, train_root_folder)
    _save_np_array(train_losses, "train_losses", network_number, train_root_folder)
    _save_np_array(validation_losses, "validation_losses", network_number, train_root_folder)


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
        map(lambda lr: (tf.keras.list_models.clone_model(model), optimizer(learning_rate=lr)),
            POSSIBLE_LEARNING_RATES))
    val_losses = list(map(lambda model_x_optimizer:
                          _train_trials(model_x_optimizer[0], train_dataset, validation_dataset, trial_epochs,
                                        model_x_optimizer[1], loss_metric_model, loss_object_model),
                          models_x_optimizers))
    models_x_optimizers_x_val_losses = zip(models_x_optimizers, val_losses)
    return min(models_x_optimizers_x_val_losses, key=lambda model: model[1])[0]


def _save_np_array(array, name, network_number, train_root_folder):
    filepath = MODEL_FILENAME_PATH(train_root_folder, network_number)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    with open('{}/{}.npy'.format(filepath, name), 'wb') as f:
        np.save(f, array)


def _save_model(model, network_number, iteration, train_root_folder):
    if not os.path.exists(MODEL_FILENAME_PATH(train_root_folder, network_number)):
        os.makedirs(MODEL_FILENAME_PATH(train_root_folder, network_number))
    model.save(MODEL_FILENAME_STRUCTURE(train_root_folder, network_number, iteration))


def _save_model_pre_train(model, network_number, root_folder):
    if not os.path.exists(MODEL_FILENAME_PATH_GEN(root_folder)):
        os.makedirs(MODEL_FILENAME_PATH_GEN(root_folder))
    model.save(MODEL_FILENAME_PATH_MODELS(root_folder, network_number))
