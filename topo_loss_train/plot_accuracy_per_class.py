import pickle
import matplotlib.pyplot as plt

from plot_results import open_pickle


def class_accuracies_epoch(acc):
    result = [[] for x in range(10)]
    for epoch in acc:
        for index in range(len(epoch)):
            result[index].append(epoch[index])

    return result


def plot_result_class(x, y, title):
    class_num = 1
    for class_acc in y:
        plt.plot(x, class_acc, label='Class ' + str(class_num))
        class_num += 1
    plt.legend(prop={'size': 6})
    plt.title(title)
    plt.show()


def plot_results(x, y, labels, title):
    for label, acc in zip(labels, y):
        plt.plot(x, acc, label=label)
    plt.legend(prop={'size': 8})
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    acc_train_class = open_pickle('AccuracyClassTrain.pkl')
    acc_test_class = open_pickle('AccuracyClassTest.pkl')
    acc_train_class_topo = open_pickle('AccuracyClassTrainTopo.pkl')
    acc_test_class_topo = open_pickle('AccuracyClassTestTopo.pkl')

    plot_range = [x for x in range(30)]

    acc_class_train = class_accuracies_epoch(acc_train_class)
    acc_class_test = class_accuracies_epoch(acc_test_class)
    acc_train_class_topo = class_accuracies_epoch(acc_train_class_topo)
    acc_test_class_topo = class_accuracies_epoch(acc_test_class_topo)

    plot_result_class(plot_range, acc_class_train, 'Class Accuracy Train')
    plot_result_class(plot_range, acc_class_test, 'Class Accuracy Test')
    plot_result_class(plot_range, acc_train_class_topo, 'Topo Class Accuracy Train')
    plot_result_class(plot_range, acc_test_class_topo, 'Topo Class Test Accuracy Test')

    acc_train = open_pickle('AccuracyTrain.pkl')
    acc_test = open_pickle('AccuracyTest.pkl')
    acc_train_topo = open_pickle('AccuracyTrainTopo.pkl')
    acc_test_topo = open_pickle('AccuracyTestTopo.pkl')

    labels = ['Acc Train', 'Acc Test', 'Acc Topo Train', 'Acc Topo Test']
    accuracies = [acc_train, acc_test, acc_train_topo, acc_test_topo]

    plot_results(plot_range, accuracies, labels, 'Accuracies per epoch')

    los = open_pickle('LossesEpochs.pkl')
    losses_topo = open_pickle('LossesEpochsTopo.pkl')
    losses_full_topo = open_pickle('LossesFullTopo.pkl')
    losses_full_none_topo = open_pickle('LossesFullNoneTopo.pkl')

    losses = [los, losses_topo, losses_full_topo, losses_full_none_topo]
    labels = ['Losses', 'Losses Topo', 'Loss Full Topo', 'Loss Full None Topo']

    plot_results(plot_range, losses, labels, 'Losses per epoch')
