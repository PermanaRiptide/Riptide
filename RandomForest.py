# Random Forest Classifier
# Derives from Classifier in Learner.py
import numpy as np
from Utilities import toIndex, fromIndex, to1ofK, from1ofK
from numpy import asarray as arr
from numpy import atleast_2d as twod
from numpy import asmatrix as mat
import Learner as L
import mltools as ml
import mltools.dtree
import matplotlib.pyplot as plt


class RandomForestClassifier(L.Classifier):

    __num_learner = 5
    __threshold = 0.5
    __ensemble = []

    def __init__(self, *args, **kwargs):
        """Constructor for Random Forest class
        Args:
          *args, **kwargs (optional): passed to train function

        Parameters
        ----------
        numLearner : number of trees in the forest.

        """
        self.classes = []

        if len(args) or len(kwargs):
            return self.train(*args, **kwargs)

    # Represents the Classifier
    def __repr__(self):
        return str(self)

    # Turns the Classifier into a string
    def __str__(self):
        return "Random Forest: [{}]"

## CORE METHODS ################################################################

    def train(self, X, Y, num_learner=5, threshold=0.5, *args, **kwargs):
        """ Train the Random Forest

        X : M x N numpy array of M data points with N features each
        Y : numpy array of shape (M,) that contains the target values for each data point
        minParent : (int)   Minimum number of data required to split a node.
        minLeaf   : (int)   Minimum number of data required to form a node
        maxDepth  : (int)   Maximum depth of the decision tree.
        nFeatures : (int)   Number of available features for splitting at each node.
        """

        self.classes = list(np.unique(Y)) if len(self.classes) == 0 else self.classes   # overload
        self.__num_learner = num_learner
        self.__threshold = threshold

        print "__classes     = ", self.classes
        print "__num_learner = ", self.__num_learner
        print "__threshold   = ", self.__threshold
        print "Single Tree param = ", kwargs

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

    def predict_soft(self,X):
        """Make soft predictions on the data in X

        Args:
          X (arr): MxN numpy array containing M data points of N features each

        Returns:
          arr : M,C array of C class probabiities for each data point
        """
        YpredTree = np.zeros((X.shape[0], 2))
        for i in range(self.__num_learner):
            YpredTree += self.__ensemble[i].predictSoft(X)
            print "iteration = ", i #keeptrack iteration

        YpredTree /= float(self.__num_learner)
        return YpredTree

