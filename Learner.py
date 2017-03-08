import numpy as np
from numpy import atleast_2d as twod
from numpy import asmatrix as arr
from Utilities import toIndex
# Adapted from mltools.base.py by Alexander Ihler with permission


# Base class for Learners
class Classifier:
    def __init__(self, *args, **kwargs):
        """
        Constructor for base classifier.
        Note: This should never be called, except by the derived classes
        """
        self.__classes = []
        # The derived classes will have a train method
        if len(args) or len(kwargs):
            return self.train(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Calls predict
        """
        return self.predict(*args, **kwargs)

    def predict(self, x):
        """
        Predicts on given data set x
        Abstract
        :param x: M x N array of M data points with N features each
        :return: None
        """
        idx = np.argmax(self.predict_soft(x), axis=1)
        return np.asarray(self.__classes)[idx]

    def predict_soft(self, x):
        """
        Predicts
        Abstract
        :param x: M x N array of M data points with N features each
        :return: None
        """
        raise NotImplementedError("Derived class has not implemented this method.")

    def err(self, x, y):
        """
        Computes the 0-1 Error on the data set x, y
        :param x: M x N array of data points, each with N features
        :param y: M x 1 array of class values for each data point in x
        :return: fraction of predict errors, or the error rate
        """
        y = arr(y)  # Create a matrix of y
        y_hat = arr(self.predict(x))  # Predict on x to get y_hat
        return np.mean(y_hat.reshape(y.shape) != y)  # Count the number of wrongs (errors)

    def nll(self, x, y):
        """
        Computes the negative log-likelihood on the data set x, y
        :param x: M x N array of data points, each with N features
        :param y: M x 1 array of class values for each data point in x
        :return: negative log-likelihood of the predictions
        """
        m, n = x.shape  # Get the rows, columns of x
        pred = arr(self.predict_soft(x))  # Perform a soft prediction of x
        pred /= np.sum(pred, axis=1, keepdims=True)  # Normalize the sum to 1
        y = toIndex(y, self.__classes)  # Get indices
        return - np.mean(np.log(pred[np.arange(m)]))  # Find the negative average of the log likelihood

    def auc(self, x, y):
        """
        Compute the area under the ROC curve on data set x, y
        :param x: M x N array of data points, each with N features
        :param y: M x 1 array of class values for each data point in x
        :return: Area under the ROC curve
        """
        if len(self.__classes) != 2:  # Check the number of classes
            raise ValueError("This method only supports binary classifications")

        try:
            # Try to make a soft prediction
            soft = self.predict_soft(x)[:, 1]
        except(AttributeError, IndexError):
            # Make a regular prediction
            soft = self.predict(x)

        n, d = twod(soft).shape  # Get the shape
        soft = soft.flatten() if n == 1 else soft.T.flatten()  # flatten the prediction

        indices = np.argsort(soft)  # Sort by score value
        y = y[indices]
        s_soft = soft[indices]

        # Compute the rank (averaged for ties) of sorted data
        dif = np.hstack(([True], np.diff(s_soft) != 0, [True]))
        r1 = np.argwhere(dif).flatten()
        r2 = r1[0:-1] + 0.5 * (r1[1:] - r1[0:-1]) + 0.5
        rank = r2[np.cumsum(dif[:-1]) - 1]

        # Number of true negatives and positives
        n0, n1 = sum(y == self.__classes[0]), sum(y == self.__classes[1])

        if n0 == 0 or n1 == 0:
            raise ValueError("Data of both class values not found")

        # Compute AUC using Mann-Whitney U statistic
        result = (np.sum(rank[y == self.__classes[1]]) - n1 * (n1 + 1.0) / 2.0) / n1 / n0
        return result

    def confusion(self, x, y):
        """
        Estimate the confusion matrix (y x y_hat) of x, y
        :param x: M x N array of data points, each with N features
        :param y: M x 1 array of class values for each data point in x
        :return: a matrix i x j, # of data from class i that were predicted as class j
        """
        y_hat = self.predict(x)  # Make a prediction
        num_classes = len(self.__classes)  # Get the number of classes

        indices = toIndex(y, self.__classes) +\
                  num_classes * (toIndex(y_hat, self.__classes) - 1)

        conf = np.histogram(indices, np.arange(1, num_classes**2 + 2))[0]
        conf = np.reshape(conf, (num_classes, num_classes))
        return np.transpose(conf)

    def roc(self, x, y):
        if len(self.__classes) != 2:  # Check the number of classes
            raise ValueError("This method only supports binary classifications")

        try:
            # Try to make a soft prediction
            soft = self.predictSoft(x)[:, 1]
        except(AttributeError, IndexError):
            # Make a regular prediction
            soft = self.predict(x)

        n, d = twod(soft).shape
        soft = soft.flatten() if n == 1 else soft.T.flatten()

        # Number of true negatives and positives
        n0 = float(np.sum(y == self.__classes[0]))
        n1 = float(np.sum(y == self.__classes[1]))

        if n0 == 0 or n1 == 0:
            raise ValueError("Data of both class values not found")

        # Sort data by score value
        indices = np.argsort(soft)
        y = y[indices]
        s_soft = soft[indices]

        # Compute false positives and true positive rates
        tpr = np.divide(np.cumsum(y[::-1] == self.__classes[1]).astype(float), n1)
        fpr = np.divide(np.cumsum(y[::-1] == self.__classes[0]).astype(float), n0)
        tnr = np.divide(np.cumsum(y == self.__classes[0]).astype(float), n0)[::-1]

        # Find ties in the sorting score
        same = np.append(np.asarray(s_soft[0:-1] == s_soft[1:]), 0)
        tpr = np.append([0], tpr[np.logical_not(same)])
        fpr = np.append([0], fpr[np.logical_not(same)])
        tnr = np.append([1], tnr[np.logical_not(same)])
        return fpr, tpr, tnr

class Regressor:
    def __init__(self, *args, **kwargs):
        """
        Constructor for base classifier.
        Note: This should never be called, except by the derived classes
        """
        # The derived classes will have a train method
        if len(args) or len(kwargs):
            return self.train(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Calls predict
        """
        return self.predict(*args, **kwargs)

    def mae(self, x, y):
        """
        Computes the Mean Absolute Error:
            AVG(|f(x_i) - y_i|)
        :param x: M x N array of data points, each with N features
        :param y: M x 1 array of class values for each data point in x
        :return: The Mean Absolute Error
        """
        y_hat = self.predict(x)
        return np.mean(np.absolute(y - y_hat.reshape(y.shape)), axis=0)

    def mse(self, x, y):
        """
        Computes the Mean Squared Error:
            AVG((f(x_i) - y_i)^2)
        :param x: M x N array of data points, each with N features
        :param y: M x 1 array of class values for each data point in x
        :return: The Mean Squared Error
        """
        y_hat = self.predict(x)
        return np.mean((y - y_hat.reshape(y.shape))**2, axis=0)

    def rmse(self, x, y):
        """
        Computes the Root Mean Squared Error:
        SQRT(AVG((f(x_i) - y_i)^2))
        :param x: M x N array of data points, each with N features
        :param y: M x 1 array of class values for each data point in x
        :return: The Root Mean Squared Error
        """
        return np.sqrt(self.mse(x, y))
