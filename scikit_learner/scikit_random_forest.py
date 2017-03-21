# Modified for learning purpose from:
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html
# Random Forest Scikit Classifier:
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

import numpy as np
import matplotlib.pyplot as plt

from sklearn import clone
from sklearn.datasets import load_iris
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.externals.six.moves import xrange
from sklearn.tree import DecisionTreeClassifier

#######################################################################################
from sklearn.metrics import roc_auc_score
import itertools
import Utilities as util
import transforms

#######################################################################################
# Helper Functions:


def transformxy(x, y):
    x = transforms.fpoly(x, 2, bias=False)
    x, p = transforms.rescale(x)
    return x, y, p


def transformx(x, p):
    x = transforms.fpoly(x, 2, bias=False)
    x, _ = transforms.rescale(x, p)
    return x

############################################################################
# Parameters
n_classes = 2
n_estimators = 25
plot_colors = "rb"
cmap = plt.cm.RdYlBu
plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.5  # step widths for coarse classifier guesses
RANDOM_SEED = 13  # fix the seed on each iteration

# Load data
# iris = load_iris()

n_training_data = 100000
############################################################################
# Get data from text
print "GENERATING DATA FROM TEXT"
my_x = np.genfromtxt("C:/Users/Kyaa/Documents/GitHub/Riptide/X_train.txt", delimiter="", max_rows=n_training_data)
my_y = np.genfromtxt("C:/Users/Kyaa/Documents/GitHub/Riptide/Y_train.txt", delimiter="", max_rows=n_training_data)
Xtest = np.genfromtxt("C:/Users/Kyaa/Documents/GitHub/Riptide/X_test.txt", delimiter="")

X, Y, p = transformxy(my_x, my_y)
x_test = transformx(Xtest,p)
print X.shape[1]
print x_test.shape[1]

xtr, xva, ytr, yva = util.splitData(X, Y, 0.9)  # Split to test and validation sets

############################################################################
# Scikit-learner model

plot_idx = 1

# models = [DecisionTreeClassifier(max_depth=None),
#           DecisionTreeClassifier(max_depth=None,max_features=2),
#           RandomForestClassifier(n_estimators=n_estimators,bootstrap=True),
#           RandomForestClassifier(n_estimators=n_estimators,bootstrap=True,max_features=2),
#           ExtraTreesClassifier(n_estimators=n_estimators),
#           AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
#                             n_estimators=n_estimators)]

class_weight = {0: 9,1:1}

models = [
RandomForestClassifier(n_estimators=n_estimators,max_features=2,warm_start=True,class_weight=class_weight),
RandomForestClassifier(n_estimators=n_estimators,max_features=2,class_weight=class_weight),
RandomForestClassifier(n_estimators=n_estimators,max_features=2)
          ]

############################################################################

my_pair = itertools.combinations([0,1,2,3,4,5,6,7,8,9,10,11,12,13],2)

my_pair = ([0,8],[0,6],[0,4],[0,10],[0,11])

for pair in my_pair:
    if pair == [0,6]: break         # run only one, can choose pair
    for model in models:
        # We only take the two corresponding features
        # X = iris.data[:, pair]
        # y = iris.target

        X = xtr
        y = ytr

        # Shuffle
        idx = np.arange(X.shape[0])
        #np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Standardize
        # mean = X.mean(axis=0)
        # std = X.std(axis=0)
        # X = (X - mean) / std

        # Train
        clf = clone(model)
        clf = model.fit(X, y)

        my_pred = clf.predict(xva)
        my_auc = roc_auc_score(yva,my_pred)

        print "my Validation AUC : ", my_auc
        # print "mean_accuracy : ", clf.score(xva, yva)

        ###################################### predict soft, put it into txt:

        YpredTree = clf.predict_proba(x_test)

        output_filename = 'scikit_results/Yhat_RF-nbags_'+str(n_estimators)+'-AUC_'+ str("%.2f" % my_auc) +'.txt'

        # np.savetxt(output_filename,
        #            np.vstack((np.arange(len(YpredTree)), YpredTree[:, 1])).T,
        #            '%d, %.2f', header='ID,Prob1', comments='', delimiter=',')

        print "Saved : ", output_filename
        #######################################################################################

        scores = clf.score(X, y)

        # Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(model)).split(".")[-1][:-2][:-len("Classifier")]
        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(len(model.estimators_))
        print( model_details + " with features", pair, "has a score of", scores )
        print ""

        #######################################################################################

# Plotting:

#         plt.subplot(3, 4, plot_idx)
#         if plot_idx <= len(models):
#             # Add a title at the top of each column
#             plt.title(model_title)
#
#         # Now plot the decision boundary using a fine mesh as input to a
#         # filled contour plot
#         x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#         y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#         xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#                              np.arange(y_min, y_max, plot_step))
#
#         # Plot either a single DecisionTreeClassifier or alpha blend the
#         # decision surfaces of the ensemble of classifiers
#         if isinstance(model, DecisionTreeClassifier):
#             Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#             Z = Z.reshape(xx.shape)
#             cs = plt.contourf(xx, yy, Z, cmap=cmap)
#         else:
#             # Choose alpha blend level with respect to the number of estimators
#             # that are in use (noting that AdaBoost can use fewer estimators
#             # than its maximum if it achieves a good enough fit early on)
#             estimator_alpha = 1.0 / len(model.estimators_)
#             for tree in model.estimators_:
#                 Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
#                 Z = Z.reshape(xx.shape)
#                 cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)
#
#         # Build a coarser grid to plot a set of ensemble classifications
#         # to show how these are different to what we see in the decision
#         # surfaces. These points are regularly space and do not have a black outline
#         xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, plot_step_coarser),
#                                              np.arange(y_min, y_max, plot_step_coarser))
#         Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
#         cs_points = plt.scatter(xx_coarser, yy_coarser, s=15, c=Z_points_coarser, cmap=cmap, edgecolors="none")
#
#         # Plot the training points, these are clustered together and have a
#         # black outline
#         for i, c in zip(xrange(n_classes), plot_colors):
#             idx = np.where(y == i)
#             plt.scatter(X[idx, 0], X[idx, 1], c=c, #label=iris.target_names[i],
#                         cmap=cmap)
#
#         plot_idx += 1  # move on to the next plot in sequence
#
# plt.suptitle("Classifiers on feature subsets of the Iris dataset")
# plt.axis("tight")
#
# plt.show()