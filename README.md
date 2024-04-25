# Topological Optimization

In order to understand the project I recommend reading my [thesis](https://sergioescalera.com/wp-content/uploads/2022/06/Saguillo_memoria.pdf). 

## Introduction

In recent years, Deep Learning (DL) models have achieved remarkable success in various tasks, including image classification, natural language processing, and speech recognition. However, one of the key challenges faced by DL models is overfitting, where the model learns to memorize the training data rather than generalize to unseen data.

## Objective

The objective of this project is to investigate the use of Topological Optimization techniques, specifically Persistent Homology (PH), to mitigate the overfitting problem in DL models. PH is a tool used in Topological Data Analysis (TDA) that captures the shape and structure of data through topological features that persist across different scales.

## Methodology

### 1. Data Preprocessing
- Preprocess the input data for the DL model, ensuring normalization and appropriate feature scaling.

### 2. DL Model Architecture
- Design a deep neural network architecture suitable for the target task, such as image classification or regression.
- Train the DL model using standard techniques, such as gradient descent-based optimization algorithms.

### 3. Topological Optimization
- Utilize Persistent Homology to analyze the topology of the DL model's decision boundary.
- Identify topological features that are prone to overfitting or capturing noise in the data.

### 4. Regularization and Optimization
- Develop regularization techniques based on topological insights to encourage smoother decision boundaries and reduce overfitting.
- Optimize the DL model parameters using the combined loss function incorporating both traditional loss terms and topological regularization terms.

### 5. Evaluation
- Evaluate the performance of the proposed approach on benchmark datasets, comparing against baseline DL models.
- Measure the generalization ability of the optimized DL models on unseen test data.
