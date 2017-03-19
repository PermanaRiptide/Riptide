import numpy as np
from Utilities import toIndex, fromIndex, to1ofK, from1ofK
from numpy import asarray as arr
from numpy import atleast_2d as twod
from numpy import asmatrix as mat
import Learner as L
import mltools as ml
import Utilities as util


def transformxy(x, y):
    x = ml.transforms.fpoly(x, 2, bias=False)
    x, p = ml.transforms.rescale(x)
    return x, y, p


def transformx(x, p):
    x = ml.transforms.fpoly(x, 2, bias=False)
    x, _ = ml.transforms.rescale(x, p)
    return x


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


def gradient_boosting():
    np.random.seed(0)

    n_training_data = 100000

    # Get data from text
    print "GENERATING DATA FROM TEXT"
    origin_x = np.genfromtxt("X_train.txt", delimiter="", max_rows=n_training_data)
    origin_y = np.genfromtxt("Y_train.txt", delimiter="", max_rows=n_training_data)
    origin_x_test = np.genfromtxt("X_test.txt", delimiter="")

    x, y, p = transformxy(origin_x, origin_y)
    x_test = transformx(origin_x_test,p)

    # x = origin_x
    # y = origin_y
    # x_test = origin_x_test
    print x.shape[1]
    print x_test.shape[1]



    print "SHUFFLING AND SPLITTING DATA"
    # x, y = util.shuffleData(x, y)  # Shuffle data
    xtr, xva, ytr, yva = util.splitData(x, y, 0.9)  # Split to test and validation sets

    nUse= 5
    mu = ytr.mean()
    dY = ytr - mu
    step = 0.5

    Pt2 = np.zeros((xtr.shape[0],))+mu
    Pv2 = np.zeros((xva.shape[0],))+mu
    Pe2 = np.zeros((x_test.shape[0],))+mu

    my_max_depth = 8
    my_min_leaf = 128
    my_nFeatures = 10

    for l in range(nUse):             # this is a lot faster than the bagging loop:
        # Better: set dY = gradient of loss at soft predictions Pt
        # Note: treeRegress expects 2D target matrix
        tree = ml.dtree.treeRegress(xtr,dY[:,np.newaxis],maxDepth=my_max_depth)
        # minLeaf = my_min_leaf ... nFeatures=my_nFeatures ... maxDepth=my_max_depth
        Pt2 += step*tree.predict(xtr)[:,0]        # predict on training data
        Pv2 += step*tree.predict(xva)[:,0]        #    and validation data
        Pe2 += step*tree.predict(x_test)[:,0]        #    and test data
        dY  -= step*tree.predict(xtr)[:,0]        # update residual for next learner

        print " {} Tr trees: MSE ~ {};  AUC - {};".format(l+1, ((ytr-Pt2)**2).mean(), auc(Pt2,ytr) )
        print " {} V  trees: MSE ~ {};  AUC - {};".format(l+1, ((yva-Pv2)**2).mean(), auc(Pv2,yva) )

    #output_filename = 'results/Yhat_gradientBoosting_119-'+str(nUse)+"-minleaf_"+str(my_min_leaf)+'.txt'
    #output_filename = 'results/Yhat_gradientBoosting_119-' + str(nUse) + "-nfeatures_" + str(my_nFeatures) + '.txt'
    output_filename = 'results/Yhat_gradientBoosting_119-' + str(nUse) + "-maxDepth_" + str(my_max_depth) + '.txt'

    np.savetxt(output_filename,
    np.vstack((np.arange(len(Pe2)), Pe2[:, ])).T,
    '%d, %.2f', header='ID,Prob1', comments='', delimiter=',');

    print "Saved : ", output_filename

    # toKaggle('Pe2.csv',Pe2)
    # print "2: GradBoost, {} trees: MSE ~ {}; AUC - {};".format(nUse, ((yva-Pv2)**2).mean(), tree.auc(Pv2,yva))

if __name__ == '__main__':
    gradient_boosting()
