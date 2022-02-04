from networks.pick_inverted import pick
import pathlib

TRAIN_ROOT_FOLDER = pathlib.Path().absolute()
NUMBER_OF_NETWORKS = 60

def pick_models():
    for i in range(NUMBER_OF_NETWORKS):
        pick(i, TRAIN_ROOT_FOLDER)

if __name__ == "__main__":
    pick_models()