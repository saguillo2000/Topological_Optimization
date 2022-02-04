from scipy.io import wavfile
import glob
import pathlib
import numpy as np
import tensorflow as tf

from constraints import BATCH_SIZE

RECORDINGS_PER_DIGIT = 300
NUMBER_OF_DIGITS = 10


def get_dataset():
    full_dataset_size = RECORDINGS_PER_DIGIT * NUMBER_OF_DIGITS
    X = get_inputs()
    y = get_labels()
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    return _generate_train_val_test_datasets(full_dataset_size, dataset)


def _generate_train_val_test_datasets(full_dataset_size, full_dataset, train_per=0.6, test_per=0.2):
    train_size = round(train_per * full_dataset_size)
    test_size = round(test_per * full_dataset_size)
    full_dataset = full_dataset.shuffle(test_size)
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    return train_dataset.batch(BATCH_SIZE), val_dataset.batch(BATCH_SIZE), test_dataset.batch(BATCH_SIZE)


def get_inputs():
    data_path = _get_data_path()
    raw_recordings = list(map(lambda digit: _get_all_digit_audios(digit, data_path), range(NUMBER_OF_DIGITS)))
    max_number_of_samples = _get_max_number_of_samples(raw_recordings)
    recordings = list(map(lambda digit_list: _pad_zeros_to_the_right(digit_list, max_number_of_samples),
                          raw_recordings))
    digit_recordings_array = list(map(lambda digit_list: np.stack(digit_list, axis=0), recordings))
    X = np.concatenate(digit_recordings_array, axis=0)
    assert X.shape[0] == RECORDINGS_PER_DIGIT * NUMBER_OF_DIGITS
    assert X.shape[1] == max_number_of_samples
    return X


def get_labels():
    y_for_digits = list(map(lambda digit: np.full(RECORDINGS_PER_DIGIT, digit), range(NUMBER_OF_DIGITS)))
    return np.concatenate(y_for_digits, axis=0)


def _get_data_path():
    dataset_root_folder = pathlib.Path().absolute()
    return '{}/data'.format(dataset_root_folder)


def _pad_zeros_to_the_right(digit_list, max_number_of_samples):
    return list(map(lambda recording: np.pad(recording, (0, max_number_of_samples - recording.shape[0]), 'constant'),
                    digit_list))


def _get_max_number_of_samples(raw_recordings):
    max_samples_per_digit = list(map(_get_max_samples_per_digit, raw_recordings))
    return max(max_samples_per_digit)


def _get_max_samples_per_digit(recordings_per_digit):
    return max(recordings_per_digit, key=lambda x: x.shape[0]).shape[0]


def _get_all_digit_audios(digit, data_path):
    digit_pattern = '{}/{}*.wav'.format(data_path, digit)
    digit_recordings = glob.glob(digit_pattern)
    assert len(digit_recordings) == RECORDINGS_PER_DIGIT
    return list(map(lambda filename: wavfile.read(filename)[1], digit_recordings))


if __name__ == "__main__":
    get_dataset()
