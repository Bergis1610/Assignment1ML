import numpy as np 
import pandas as pd 

# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self, alpha=0.04, iter=3000):        
        # learning rate
        self.alpha = alpha
        # Iterations
        self.iter = iter
        
    
                                   
    def fit(self, X, y):
        self.rows, self.cols = X.shape
        
        self.weights = np.zeros(self.cols)
        self.bias = 0
        self.X = X
        self.y = y
        learningRate = self.alpha
        
        # Optimize the function
        for i in range(self.iter):
            theta = sigmoid((self.X.dot(self.weights) + self.bias))
    
            dbias = np.mean(theta - self.y)
            dweights = (1/len(self.X))*np.matmul(self.X.transpose(), (theta - self.y))
            
            self.bias = self.bias - (learningRate * dbias)
            self.weights = self.weights - (learningRate * dweights)
                      
        
    def predict(self, X):
        
        # pred = 1 / (1 + np.exp(- (X.dot(self.weights.transpose()) + self.bias)))
        pred = sigmoid((X.dot(self.weights) + self.bias))
   
        pred = np.where(pred < 0.5, 0, 1)
        return pred
        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy

    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions

    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()


#Loss
def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 

    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions

    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(y_true * np.log(y_pred) + (1 - y_true) * (np.log(1 - y_pred)))


def sigmoid(x):
    """
    Applies the logistic function element-wise

    Hint: highly related to cross-entropy loss 

    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.

    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

        

 