"""
Random Forest Classifier
"""

import numpy as np
import Learner as L
import mltools as ml
import Utilities as util

################################################################################
# Helper Functions:


def auc(soft, Y):
    """Manual AUC function for applying to soft prediction vectors"""
    indices = np.argsort(soft)  # sort data by score value
    Y = Y[indices]
    sorted_soft = soft[indices]

    # compute rank (averaged for ties) of sorted data
    dif = np.hstack(([True], np.diff(sorted_soft) != 0, [True]))
    r1 = np.argwhere(dif).flatten()
    r2 = r1[0:-1] + 0.5 * (r1[1:] - r1[0:-1]) + 0.5
    rnk = r2[np.cumsum(dif[:-1]) - 1]

    # number of true negatives and positives
    n0, n1 = sum(Y == 0), sum(Y == 1)

    # compute AUC using Mann-Whitney U statistic
    result = (np.sum(rnk[Y == 1]) - n1 * (n1 + 1.0) / 2.0) / n1 / n0
    return result


class RandomForestClassifier(L.Classifier):

    __num_learner = 5
    __threshold = 0.5
    __ensemble = []
    __confidence_level = None
    __isBoosted = False
    __time_to_train = 5

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

# CORE METHODS ################################################################

    def train(self, X, Y, Xtest = None, isboosted=False,time_to_train=10,num_learner=5, threshold=0.5, *args, **kwargs):
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
        self.__isBoosted = isboosted

        # print "__classes     = ", self.classes
        print "__num_learner = ", self.__num_learner
        print "__threshold   = ", self.__threshold
        print "__isBoosted   = ", self.__isBoosted
        print "Single Tree param = ", kwargs


#########################################################################################

        # With Boosting :

        xtr, xva, ytr, yva = util.splitData(X, Y, 0.9)  # Split to test and validation sets
        x_test = Xtest

        if self.__isBoosted:
            print "Boosted!"
            print x_test.shape[0]

            YpredTree = np.zeros((x_test.shape[0], ))

            # Embedded single tree parameters:
            # my_min_leaf = 128
            # print "my_min_leaf : ", my_min_leaf
            # my_max_depth = 4
            # print "my_max_depth : ", my_max_depth
            # my_n_features = 2
            # print "my_n_features : ", my_n_features

            for i in range(self.__num_learner):
                print "my_iteration : ", i+1

                Xi, Yi = ml.bootstrapData(xtr, ytr)     # (xtr, ytr)
                # save ensemble member "i" in a cell array

                nUse = time_to_train
                mu = Yi.mean()
                dY = Yi - mu
                step = 0.5

                Pt2 = np.zeros((Xi.shape[0],)) + mu
                Pv2 = np.zeros((xva.shape[0],)) + mu
                Pe2 = np.zeros((x_test.shape[0],)) + mu

                tree = None
                for l in range(nUse):  # this is a lot faster than the bagging loop:
                    print "my_boosting : ", l + 1
                    # Better: set dY = gradient of loss at soft predictions Pt
                    # Note: treeRegress expects 2D target matrix
                    tree = ml.dtree.treeRegress(Xi, dY[:, np.newaxis], *args, **kwargs)  # ,minLeaf=my_min_leaf train and save learner
                    Pt2 += step * tree.predict(Xi)[:, 0]  # predict on training data
                    Pv2 += step * tree.predict(xva)[:, 0]  # and validation data
                    # Pe2 += step * tree.predict(x_test)[:, 0]  # and test data
                    dY -= step * tree.predict(Xi)[:, 0]  # update residual for next learner

                    print " {} Tr trees: MSE ~ {};  AUC - {};".format(l + 1, ((ytr - Pt2) ** 2).mean(), auc(Pt2, ytr))
                    print " {} V  trees: MSE ~ {};  AUC - {};".format(l + 1, ((yva - Pv2) ** 2).mean(), auc(Pv2, yva))
                exit()
                self.__ensemble.append(tree)
                YpredTree += Pe2

            YpredTree /= float(self.__num_learner)
            self.__confidence_level = YpredTree
#########################################################################################
        else:
            # Without Boosting
            for i in range(self.__num_learner):
                Xi, Yi = ml.bootstrapData(X, Y)
                # save ensemble member "i" in a cell array
                self.__ensemble.append(ml.dtree.treeClassify(Xi, Yi, *args, **kwargs))


#########################################################################################

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
        self.__confidence_level = YpredTree
        return YpredTree

    def get_confidence(self):
        return self.__confidence_level

