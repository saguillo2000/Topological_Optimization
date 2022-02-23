import pickle
import matplotlib.pyplot as plt
import plotly.express as px


def open_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def flat_list(losses):
    return [item for sublist in losses for item in sublist]


if __name__ == '__main__':
    none_topo_losses = open_pickle('losses_epochs.pkl')
    topo_losses = open_pickle('topo_losses_epochs.pkl')
    validations = open_pickle('validations_epochs.pkl')

    validations_topo = []
    validations_none_topo = []
    for index, tuple_ in enumerate(validations):
        val_none_topo = tuple_[0]
        val_topo = tuple_[1]

        validations_topo.append(sum(val_topo) / len(val_topo))
        validations_none_topo.append(sum(val_none_topo) / len(val_none_topo))

        print('-------------------')
        print('In epoch: ', index)
        print('-------------------')
        print('Topological losses: ',sum(val_topo) / len(val_topo))
        print('None topological losses: ',sum(val_none_topo) / len(val_none_topo))

    flat_topo_losses = flat_list(topo_losses)
    flat_none_topo_losses = flat_list(none_topo_losses)
    flat_val_topo = validations_topo
    flat_val_none_topo = validations_none_topo

    # Create count of the number of epochs
    epoch_count = range(1, len(flat_val_topo) + 1)

    # Visualize loss history
    plt.plot(epoch_count, flat_val_topo, 'r--')
    plt.plot(epoch_count, flat_val_none_topo, 'b--')
    plt.legend(['Topo Losses', 'None Topo Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([1.5, 2.9])
    plt.show()
