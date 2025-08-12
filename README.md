# Neural Network from Scratch: Classifying Letters A, B, and C

## Overview
This project demonstrates the implementation of a simple feedforward neural network using only NumPy to classify synthetic 5x6 binary images of the letters A, B, and C. The network is trained using custom backpropagation and visualizes loss and accuracy over epochs.

## Approach & Methodology
- Data Preparation:
  - Created binary (0/1) 5x6 pixel patterns for the letters A, B, and C.
  - Flattened each pattern to a 1D array (length 30) and stacked them as the training set.
  - Used one-hot encoding for the labels: [1,0,0] for A, [0,1,0] for B, [0,0,1] for C.

- Neural Network Architecture:
  - Input layer: 30 neurons (one per pixel).
  - Hidden layer: 10 neurons, sigmoid activation.
  - Output layer: 3 neurons (one per class), softmax activation.
  - Weights and biases initialized randomly.

- Training:
  - Used cross-entropy loss and tracked accuracy.
  - Implemented feedforward and backpropagation manually using matrix operations.
  - Updated weights and biases using gradient descent.
  - Trained for 5000 epochs, tracking loss and accuracy at each epoch.

- Visualization:
  - Plotted loss and accuracy curves over epochs using matplotlib.
  - Displayed the input images and their predicted classes after training.

## Analysis Process
1. Feedforward:
   - Input is passed through the network using matrix multiplications and activation functions.
   - Output probabilities are computed using softmax.
2. Loss Calculation:
   - Cross-entropy loss is computed between predicted and true labels.
3. Backpropagation:
   - Gradients are calculated for each layer using the chain rule.
   - Weights and biases are updated to minimize the loss.
4. Evaluation:
   - Accuracy is computed by comparing predicted and true classes.
   - Visualizations help monitor training progress and final predictions.

## Key Findings
- The neural network quickly learns to classify the three patterns with high accuracy, even with a small dataset.
- Loss decreases and accuracy increases rapidly, demonstrating effective learning.
- The model is able to correctly predict the class of each input pattern after training.
- This project provides hands-on experience with the core building blocks of neural networks: weight initialization, activation functions, loss computation, and gradient descent.

## How to Run
1. Open the notebook `Neural_Network_From_Scratch.ipynb` in Jupyter or VS Code.
2. Run all cells in order to train the model and visualize results.
3. Review the plots and prediction outputs for analysis.

Author: Umang Dixit
Date: 12 August 2025
