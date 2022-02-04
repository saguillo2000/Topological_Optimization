# Toy networks project
The aim of this project is to generate an easy-to-analyze dataset for solving the problem *"Predicting the generalization gap"*. 

This dataset consists on 3 different tasks. Each task contains a usual dataset (images, text and audio, already preprocessed) and a lot of networks trained with that dataset. For each network, three stages of the training have been selected: 

 1. `nice_fitting.h5` The network trained at the epoch that minimizes the absolute distance between the validation loss function and the train loss function.
 2. `under_fitting.h5` The network, in an epoch previous to the `nice_fitting` one, whose validation loss is the closest to the absolute value of (loss initial random configuration - loss best model)/2
 3. `over_fitting.h5`. The network trained X epochs after the nice_fitting epoch is found. (X is a configurable parameter)

The networks are generated using the shape of a probabilistic distribution. For the ones that are already generated in the project, we have used the [Pareto distribution]. (https://en.wikipedia.org/wiki/Pareto_distribution) (but it can be changed in the `train.py ` script). Moreover, in order to analyze the evolution of the networks, we have included, for each complete network shape, a sequence of networks based on the complete one that incrementally add a new hidden layer to the right of the network.

Regarding the training process, we have split every dataset into train (60% of the dataset), validation (20%) and test (20%) sets. Every networks has been trained with the `Adam` optimizer using Keras. In order to select the best learning rate, we have used the validation dataset to get the one that gave the best result for the first Y epochs (configurable parameter, as well as the learning rates to explore). `SGD` and other optimizers can be used. In order to assure the proper behavior of the training algorithm, the dataset should be split in batches (by default, the project uses batches of 32 elements, but it can be changed). In order to generate the networks, to train them, and to pick the three interesting iterations, a recommended approach is to execute a code similar to the ones found in each dataset folder, in the `train.py` file.


As we mentioned previously, we selected three different datasets, one for each task. The tasks consist on:

 - Text task: Sentiment analysis: We have used the `Large Movie Review Dataset v1.0` dataset. published on http://www.aclweb.org/anthology/P11-1015. In this task the networks try to detect if a IMDB review is positive or negative. For this task, we added a word embedding tensorflow layer to optimize the representation of the text. Moreover we have used the standard pipeline to convert text into a vector representation in Tensorflow. See https://www.tensorflow.org/tutorials/keras/text_classification
 - Image task: Image recognition: We have used the CIFAR10 dataset in the usual classification problem.
 - Sound task: We have used this dataset:  [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset). In order to standarize the input vectors, we have added silence ('0' elements) at the end of the recording until complete the input vector. The size of the input vector is equal to the maximal size of the input vector for the whole dataset.

The algorithm that generates automatically the networks can be found at the file: `networks/MLP/mlp_generation.py`. The algorithm that trains the networks can be found at `networks/train.py` and finally the algorithm that pick the 3 desired iterations for each network can be found at `networks/pick_and_clean.py'`

All the configurable variables can be found in the file `constraints.py`

Every dataset is trained in the `train.py` script inside its respective folder. Moreover, the dataset is generated and preprocessed at the file `dataset.py`, again, inside its respective folder.
