import pickle
import matplotlib.pyplot as plt

from plot_results import open_pickle


def class_accuracies_epoch(acc):
    result = [[] for x in range(10)]
    for epoch in acc:
        for index in range(len(epoch)):
            result[index].append(epoch[index])

    return result


if __name__ == '__main__':
    acc_train = open_pickle('AccuracyClassTrain.pkl')
    acc_test = open_pickle('AccuracyClassTest.pkl')

    print()

    print(acc_test)

    plot_range = [x for x in range(20)]
    acc_class_train = class_accuracies_epoch(acc_train)
    acc_class_test = class_accuracies_epoch(acc_test)

    class_num = 1
    for epoch in acc_class_train:
        plt.plot(plot_range, epoch, label='Class '+str(class_num))
    plt.legend()
    plt.show()

    class_num = 1
    for epoch in acc_class_test:
        plt.plot(plot_range, epoch, label='Class ' + str(class_num))
    plt.legend()
    plt.show()
