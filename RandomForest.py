# Random Forest Classifier
# Derives from Classifier in Learner.py
import numpy as np
from Utilities import toIndex, fromIndex, to1ofK, from1ofK
from numpy import asarray as arr
from numpy import atleast_2d as twod
from numpy import asmatrix as mat
from Learner import Classifier as Cl

import mltools as ml
import mltools.dtree
import matplotlib.pyplot as plt

class RandomForestClassifier():

    __num_learner = 5
    __threshold = 2.0
    __ensemble = []

    def __init__(self, num_learner=5, threshold=2.0, *args, **kwargs):
        """Constructor for Random Forest class
        Args:
          *args, **kwargs (optional): passed to train function

        Parameters
        ----------
        numLearner : number of trees in the forest.

        """
        self.__num_learner = num_learner
        self.__threshold = threshold
        if len(args) or len(kwargs):
            return self.train(*args, **kwargs)


## CORE METHODS ################################################################

    def train(self, X, Y, *args,**kwargs):
        """ Train the Random Forest

        X : M x N numpy array of M data points with N features each
        Y : numpy array of shape (M,) that contains the target values for each data point
        minParent : (int)   Minimum number of data required to split a node.
        minLeaf   : (int)   Minimum number of data required to form a node
        maxDepth  : (int)   Maximum depth of the decision tree.
        nFeatures : (int)   Number of available features for splitting at each node.
        """
        for i in range(self.__num_learner):
            Xi, Yi = ml.bootstrapData(X, Y)
            # save ensemble member "i" in a cell array
            self.__ensemble.append(ml.dtree.treeClassify(Xi, Yi, *args, **kwargs))


    def predict(self,X):
        """Make predictions on the data in X

        Args:
          X (arr): MxN numpy array containing M data points of N features each

        Returns:
          arr : M, or M,1 vector of target predictions
        """
        tree_prediction = []
        majority_prediction = []
        for j in range(self.__num_learner):
            tree_prediction.append(self.__ensemble[j].predict(X))

        for i in range(X.shape[0]):
            if self.__num_learner == 1:
                majority_prediction.append(tree_prediction[0][i])
            else:
                majority_prediction.append(1 if [item[i]
                    for item in tree_prediction[0:self.__num_learner]].count(1) > self.__num_learner / self.__threshold else 0)

        return majority_prediction

    def predictSoft(self,X):
        """Make soft predictions on the data in X

        Args:
          X (arr): MxN numpy array containing M data points of N features each

        Returns:
          arr : M,C array of C class probabiities for each data point
        """
        YpredTree = np.zeros((Xtest.shape[0], 2))
        for i in range(self.__num_learner):
            YpredTree += ensemble[i].predictSoft(Xtest)
            # print i #keeptrack iteration

        YpredTree /= float(self.__num_learner)
        return YpredTree

