import pickle
import matplotlib.pyplot as plt
import plotly.express as px


def open_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def flat_list(losses):
    return [item for sublist in losses for item in sublist]


def split_values(validations):
    topo_values = []
    none_topo_values = []
    for index, tuple_ in enumerate(validations):
        val_none_topo = tuple_[0]
        val_topo = tuple_[1]

        topo_values.append(sum(val_topo) / len(val_topo))
        none_topo_values.append(sum(val_none_topo) / len(val_none_topo))

        print('-------------------')
        print('In epoch: ', index)
        print('-------------------')
        print('Topological values: ', sum(val_topo) / len(val_topo))
        print('None topological values: ', sum(val_none_topo) / len(val_none_topo))

    return topo_values, none_topo_values


if __name__ == '__main__':
    none_topo_losses = open_pickle('losses_epochs.pkl')
    topo_losses = open_pickle('topo_losses_epochs.pkl')
    validations = open_pickle('validations_epochs.pkl')
    accuracies = open_pickle('validations_accuracies.pkl')

    validations_topo, validations_none_topo = split_values(validations)
    accuracies_topo, accuracies_none_topo = split_values(accuracies)

    flat_topo_losses = flat_list(topo_losses)
    flat_none_topo_losses = flat_list(none_topo_losses)

    flat_acc_topo = accuracies_topo
    flat_acc_none_topo = accuracies_none_topo

    # Create count of the number of epochs
    epoch_count = range(1, len(flat_acc_topo) + 1)

    # Visualize loss history
    plt.plot(epoch_count, flat_acc_topo, 'r--')
    plt.plot(epoch_count, flat_acc_none_topo, 'b--')
    plt.legend(['Topo Accuracy', 'None Topo Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # plt.ylim([0.10, 0.15])
    plt.show()

    epoch_count = range(1, len(validations_topo) + 1)

    # Visualize loss history
    plt.plot(epoch_count, validations_topo, 'r--')
    plt.plot(epoch_count, validations_none_topo, 'b--')
    plt.legend(['Topo Losses', 'None Topo Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.ylim([0.10, 0.15])
    plt.show()
