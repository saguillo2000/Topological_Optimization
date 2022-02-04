BATCH_SIZE = 32

MAX_NEURONS_FOR_NETWORK = 1000
QT_BATCHES = 10
MAX_HIDDEN_LAYERS = 6

MAX_WORD_FEATURES = 250
EMBEDDING_DIM = 16

EPOCHS = 150  # Production 150
SAVING_MODEL_BATCHES_ITERATIONS_PERIOD = 1  # It indicates how many iterations are as gap of the saved models
EPOCH_ITERATIONS_TO_INCREMENT_ITERATION_BATCHES = 1  # Every these epochs the number of the batches in each iteration increase
TRIAL_EPOCHS = 4  # TODO in production 5
POSSIBLE_LEARNING_RATES = [0.01, 0.001, 0.0001, 0.00001, 0.000001]

SKIP_INDICES_FOR_OVERFITTED = 10


def MODEL_FILENAME_PATH(root_folder, model_number):
    return '{}/models/{}'.format(root_folder, model_number)


def MODEL_FILENAME_PATH_GEN(root_folder):
    return '{}/models'.format(root_folder)


def MODEL_FILENAME_STRUCTURE(root_folder, model_number, epoch_number):
    return '{}/models/{}/{}.h5'.format(root_folder, model_number, epoch_number)


def MODEL_FILENAME_PATH_MODELS(root_folder, model_number):
    return '{}/models/{}.h5'.format(root_folder, model_number)
