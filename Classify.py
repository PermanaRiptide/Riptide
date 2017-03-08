# By: Joseph Park, Kevin Permana
# Adapted from Alexander Ihler's CS 178 - Data Mining/Machine Learning course at UCI
import numpy as np  # For general matrix and math operations
import Utils


class Classify(object):
    def __init__(self, classes=[]):
        """
        Constructor for Classify
        :param classes: The classes for the this classifer. DEFAULT: No classes
        """
        self.__cl = classes

    ################
    # CORE METHODS #
    ################
    def train(self, x, y):
        """
        Train the Classifier on the data set x, y.
        Note: This is an abstract method.
        :param x: M x N matrix of M data points with N features each
        :param y: M x 1 matrix of M classes, one for each data point in x
        """
        # This is training the classifier
        # This will be different based on the classifier
        raise NotImplementedError

    def predict(self, x):
        """
        Predict the classes of the data set x
        :param x: M x N matrix of M data points with N features each
        :return: M x 1 matrix of M predicted classes, one for each data point in x
        """
        # Try to predict (hard boundary) the class of each point in x
        index = np.argmax(self.predict_soft(x), axis=1)  # Find the most likely class
        return np.asarray(self.__cl)[index]  # Convert to the most likely class value

    def predict_soft(self, x):
        """
        Predict the classes of the data set x, along with the confidence of each prediction
        NOTE: This is an abstract method.
        :param x: M x N matrix of M data points with N features each
        """
        # Try to make a soft prediction
        raise NotImplementedError

    ##################
    # Loss Functions #
    ##################
    def error(self, x, y):
        """
        Calculate the Classification Error of the data set x, y
        :param x: M x N matrix of M data points with N features each
        :param y: M x 1 matrix of M classes, one for each data point in x
        :return: A float representing the Classification Error Rate
        """
        # Find the error rate on the data set
        # np.asmatrix(y) turns y (array) into an np.matrix object
        # y.reshape(x) turns y into an array of x's shape (dimensions)
        y = np.asmatrix(y)  # Ensure that y is a matrix
        y_hat = np.asmatrix(self.predict(x))  # We want hard predictions so that we can get the Classification Error
        # Find the number of y_hat's that do not match y's.
        # Reshape to ensure that y_hat and y are the same shape before comparing each element
        return np.mean(y_hat.reshape(y.shape) != y)

    def nll(self, x, y):
        """
        Calculate the Negative Log-Likelihood of the data set x, y
        :param x: M x N matrix of M data points with N features each
        :param y: M x 1 matrix of M classes, one for each data point in x
        :return: A float representing the Negative Log-Likelihood
        """
        # Get the dimensions of x
        m, n = x.shape
        y_hat = np.asmatrix(self.predict_soft(x))  # Get soft predictions

        # Normalize the predictions (By dividing by the sum)
        y_hat /= np.sum(y_hat, axis=1, keepdims=True)  # Divide the predictions by the total sum of predictions
        y = Utils.toIndex(y, self.__cl)  # Get the classes
        # Calculate the negative log-likelihood
        return -np.mean(np.log(y_hat[np.arange(m), y]))

    # TODO: auc, roc, confusion
