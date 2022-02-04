import tensorflow as tf
import os
from functools import partial

import numpy as np

from constraints import MODEL_FILENAME_PATH, SKIP_INDICES_FOR_OVERFITTED


@tf.function
def compute_predictions(inputs, model):
    return model(inputs)


def pick_and_clean_models(train_dataset, validation_dataset, initial_network, model_number, train_root_folder,
                          loss_object_model, loss_metric_model):
    model_path = MODEL_FILENAME_PATH(train_root_folder, model_number)
    if not os.path.exists(model_path):
        raise Exception("The model folder does not exist")
    models_files = [model_file for model_file in os.listdir(model_path) if model_file.endswith('.h5')]
    models_iterations = [int(model_file[:-3]) for model_file in models_files]
    models = [tf.keras.models.load_model(os.path.join(model_path, model_file), compile=False)
              for model_file in models_files]
    models_x_iterations = list(zip(models, models_iterations))
    best_model_x_iteration = _pick_best_model(models_x_iterations, train_dataset, validation_dataset, loss_object_model,
                                              loss_metric_model)
    best_model_x_iteration[0].save('{}/nice_fitting_it_{}.h5'.format(model_path, best_model_x_iteration[1]))
    idx_best_model = models.index(best_model_x_iteration[0])
    if idx_best_model != 0:
        underfitted_model = _pick_underfitted_model(best_model_x_iteration[0], models_x_iterations[:idx_best_model],
                                                    validation_dataset,
                                                    initial_network,
                                                    loss_object_model, loss_metric_model)
        underfitted_model[0].save('{}/under_fitting_it_{}.h5'.format(model_path, underfitted_model[1]))
    if idx_best_model != (len(models) - 1):
        idx_overfitted_model = min(len(models), models.index(best_model_x_iteration[0]) + SKIP_INDICES_FOR_OVERFITTED)
        overfitted_model = models_x_iterations[idx_overfitted_model]
        overfitted_model[0].save('{}/over_fitting_it_{}.h5'.format(model_path, overfitted_model[1]))
    _remove_not_valuable_models(model_path)


def _remove_not_valuable_models(model_path):
    def file_to_delete(file):
        return not ("fitting" in file or file.endswith(".npy"))

    files = [f for f in os.listdir(model_path) if file_to_delete(f)]
    for file in files:
        os.remove(os.path.join(model_path, file))


def _pick_underfitted_model(best_model, models_x_iterations, validation_dataset, initial_network, loss_object_model,
                            loss_metric_model):
    compute_loss = partial(_compute_loss_for_validation,
                           validation_dataset=validation_dataset,
                           loss_object_model=loss_object_model,
                           loss_metric_model=loss_metric_model)
    loss_of_best_model = compute_loss(best_model)
    loss_of_initial_network = compute_loss(initial_network)
    desired_underfitted_loss = (np.abs(loss_of_initial_network - loss_of_best_model)) / 2
    losses_of_models = map(lambda model_x_iteration: compute_loss(model_x_iteration[0]), models_x_iterations)
    diff_losses_and_desired = map(lambda loss: np.abs(desired_underfitted_loss - loss), losses_of_models)
    models_x_iterations_x_diff = list(zip(models_x_iterations, diff_losses_and_desired))
    return min(models_x_iterations_x_diff, key=lambda model_x_iteration_x_diff: model_x_iteration_x_diff[1])[0]


def _compute_loss_for_validation(model, validation_dataset, loss_object_model, loss_metric_model):
    loss_object = loss_object_model
    validation_loss = loss_metric_model(name='pick_validation_loss')
    for validation_inputs, validation_labels in validation_dataset:
        predictions = compute_predictions(validation_inputs, model)
        v_loss = loss_object(validation_labels, predictions)
        validation_loss(v_loss)
    loss_of_model = validation_loss.result().numpy()
    validation_loss.reset_states()
    return loss_of_model


def _pick_best_model(models_x_iterations, train_dataset, validation_dataset, loss_object_model, loss_metric_model):
    diff_of_losses = map(lambda model_x_iteration: _compute_difference_between_training_and_validation_losses(
        model_x_iteration[0],
        train_dataset,
        validation_dataset,
        loss_object_model,
        loss_metric_model),
                         models_x_iterations)
    models_x_diff = list(zip(models_x_iterations, diff_of_losses))
    return min(models_x_diff, key=lambda model_x_diff: model_x_diff[1])[0]


def _compute_difference_between_training_and_validation_losses(model, train_dataset, validation_dataset,
                                                               loss_object_model, loss_metric_model):
    loss_object = loss_object_model
    train_loss = loss_metric_model(name='pick_train_loss')
    validation_loss = loss_metric_model(name='pick_validation_loss')
    for inputs, labels in train_dataset:
        predictions = compute_predictions(inputs, model)
        t_loss = loss_object(labels, predictions)
        train_loss(t_loss)
    for validation_inputs, validation_labels in validation_dataset:
        predictions = compute_predictions(validation_inputs, model)
        v_loss = loss_object(validation_labels, predictions)
        validation_loss(v_loss)
    difference = np.abs(train_loss.result().numpy() - validation_loss.result().numpy())
    train_loss.reset_states()
    validation_loss.reset_states()
    return difference
