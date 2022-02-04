from sklearn.linear_model import LinearRegression
import numpy as np
from sympy.solvers import solve
from sympy import Symbol
import os
from shutil import copyfile

from constraints import MODEL_FILENAME_PATH


def pick(model_number, root_folder):
    model_path = MODEL_FILENAME_PATH(root_folder, model_number)
    if not os.path.exists(model_path):
        raise Exception("The model folder does not exist")
    train_losses, validation_losses = _get_losses(model_path)
    reg_train, reg_validation = _get_regression_models(train_losses, validation_losses)
    saved_iterations = _get_saved_iterations(model_path)
    nice_fitting_iteration = _get_nice_fitting_iteration(saved_iterations, reg_train, reg_validation)
    under_fitting_iteration = _get_under_fitting_iteration(saved_iterations, reg_validation, nice_fitting_iteration)
    over_fitting_iteration = _get_over_fitting_iteration(saved_iterations, nice_fitting_iteration)
    _save_iteration_with_name('nice_fitting', nice_fitting_iteration, model_path)
    _save_iteration_with_name('under_fitting', under_fitting_iteration, model_path)
    _save_iteration_with_name('over_fitting', over_fitting_iteration, model_path)

def _save_iteration_with_name(name, iteration_number, model_path):
    new_model_file = '{}/{}_{}.h5'.format(model_path, name, iteration_number)
    old_model_file = '{}/{}.h5'.format(model_path, iteration_number)
    copyfile(old_model_file, new_model_file)


def _get_losses(model_path):
    train_losses = np.load('{}/train_losses.npy'.format(model_path))
    validation_losses = np.load('{}/validation_losses.npy'.format(model_path))
    return train_losses, validation_losses


def _get_saved_iterations(model_path):
    def model_condition(model_file):
        return (model_file.endswith('.h5')) and ('fitting' not in model_file)

    if not os.path.exists(model_path):
        raise Exception("The model folder does not exist")
    model_files = (model_file for model_file in os.listdir(model_path) if model_condition(model_file))
    model_iterations = [int(model_file[:-3]) for model_file in model_files]
    return model_iterations


def _get_regression_models(train_losses, validation_losses):
    number_of_iterations_per_model = len(train_losses)
    x = np.array(list(range(1, number_of_iterations_per_model + 1)))
    inverted_x = 1/x  # For 1/x
    log_x = np.log(x)
    X = np.array(list(zip(inverted_x, log_x))) # We will train models of the type a(1/x) + bln(x) + c
    reg_train = LinearRegression().fit(X, train_losses)
    reg_validation = LinearRegression().fit(X, validation_losses)
    return reg_train, reg_validation


def _get_nice_fitting_iteration(iteration_numbers, reg_train, reg_validation):
    distances = map(lambda it_num:
                    _compute_distance_between_losses(it_num, reg_train, reg_validation),
                    iteration_numbers)
    iterations_x_distances = zip(iteration_numbers, distances)
    nice_fitting_iteration, _ = min(iterations_x_distances, key=lambda it_x_dist: it_x_dist[1])
    return nice_fitting_iteration


def _get_under_fitting_iteration(iteration_numbers, reg_validation, nice_fitting_iteration):
    iteration_numbers = list(filter(lambda iteration: iteration < nice_fitting_iteration, iteration_numbers))
    iteration_numbers.sort()
    if len(iteration_numbers) == 0:
        return -1
    nice_fitting_extrapolation = _compute_extrapolated_loss_for_iteration(nice_fitting_iteration, reg_validation)
    first_iteration_extrapolation = _compute_extrapolated_loss_for_iteration(iteration_numbers[0], reg_validation)
    desired_loss_diff = np.abs(first_iteration_extrapolation - nice_fitting_extrapolation) / 2
    extrapolations_diffs = list(map(lambda it_num:
                                    np.abs(_compute_extrapolated_loss_for_iteration(it_num,
                                                                                    reg_validation) - nice_fitting_extrapolation),
                                    iteration_numbers))
    diffs = list(map(lambda extrapolation_diff: np.abs(extrapolation_diff - desired_loss_diff), extrapolations_diffs))
    it_diffs = list(zip(iteration_numbers, diffs))
    return min(it_diffs, key=lambda it_diff: it_diff[1])[0]


def _get_over_fitting_iteration(iteration_numbers, nice_fitting_iteration, skipped_iterations=100):
    iteration_numbers.sort()
    nice_fitting_idx = iteration_numbers.index(nice_fitting_iteration)
    if nice_fitting_idx == (len(iteration_numbers) - 1):
        return -1
    over_fitting_idx = min(len(iteration_numbers) - 1, nice_fitting_idx + skipped_iterations)
    return iteration_numbers[over_fitting_idx]


def _compute_extrapolated_loss_for_iteration(it_num, reg):
    input_to_regression = np.array([[(1/(it_num + 1)), np.log(it_num + 1)]])
    return reg.predict(input_to_regression)


def _compute_distance_between_losses(it_num, reg_train, reg_validation):
    input_to_regression = np.array([[(1/(it_num + 1)), np.log(it_num + 1)]])
    return np.abs(reg_train.predict(input_to_regression) - reg_validation.predict(input_to_regression))

