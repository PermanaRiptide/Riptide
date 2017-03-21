"""
Random Forest Driver
"""

import RandomForest as rf
import numpy as np
import matplotlib.pyplot as plt
import Utilities as util
import mltools as ml

################################################################################
# Helper Functions:


def transformxy(x, y):
    x = ml.transforms.fpoly(x, 2, bias=False)
    x, p = ml.transforms.rescale(x)
    return x, y, p


def transformx(x, p):
    x = ml.transforms.fpoly(x, 2, bias=False)
    x, _ = ml.transforms.rescale(x, p)
    return x
################################################################################


def testRandomForest():

    # np.random.seed(0)  # Set seed to get deterministic results
    n_training_data = 100000

    # Get data from text
    print "GENERATING DATA FROM TEXT"
    x = np.genfromtxt("X_train.txt", delimiter="", max_rows=n_training_data)
    y = np.genfromtxt("Y_train.txt", delimiter="", max_rows=n_training_data)
    x_test = np.genfromtxt("X_test.txt", delimiter="")

    # x , y , p = transformxy(origin_x, origin_y)
    # print x.shape[1]
    # print y
    # print np.unique(y)
################################################################################
    # print "SHUFFLING AND SPLITTING DATA"
    # x, y = util.shuffleData(x, y)  # Shuffle data
    xtr, xva, ytr, yva = util.splitData(x, y, 0.9)  # Split to test and validation sets
################################################################################
    print "TRAINING RANDOM FOREST"

    my_threshold = [0.5]
    my_allowed_features = [2]
    my_nBags = [10]
    boosted_status = False

################################################################################
    for nBags in my_nBags:
        for threshold in my_threshold:
            for number_of_features_allowed in my_allowed_features:

                ######################################################################################
                # Train a boosted random forest model :
                my_random_forest = \
                    rf.RandomForestClassifier(x, y,
                                              Xtest=x_test, # for boosted soft prediction
                                              isboosted=boosted_status, # boosted status, default = False
                                              num_learner=nBags, threshold=threshold, # Random Forest params
                                              nFeatures=number_of_features_allowed  # dtree paremeters
                                              )  # nFeatures=number_of_features_allowed

                #############################################################
                # Train a original random forest model :
                # my_random_forest = \
                #    rf.RandomForestClassifier(xtr, ytr, nBags, threshold)  # nFeatures=number_of_features_allowed

                ######################################################################################

                # print "my_threshold               = ", threshold
                # print "number_of_features_allowed = ", number_of_features_allowed

                etr = my_random_forest.err(xtr, ytr)
                print "Training Error: ", etr

                eva = my_random_forest.err(xva, yva)
                print "Validation Error: ", eva

                atr = my_random_forest.auc(xtr, ytr)
                print "Training AUC: ", atr

                ava = my_random_forest.auc(xva, yva)
                print "Validation AUC = ", ava

    ######################################################################################
                print "------------------------------------------------------"

                # original random forest:
                # YpredTree = my_random_forest.predict_soft(x_test)

                # for boosted forest:
                YpredTree = my_random_forest.get_confidence()

    ######################################################################################
                # save confidence to txt:

                output_filename = ""

                if boosted_status:
                    output_filename = 'results/Yhat_rf_-' + str(n_training_data) \
                                      + '-nbags_' + str(nBags) \
                                      + '-nfeat_' + str(number_of_features_allowed) \
                                      + '-thrs_' + str(threshold) + '.txt'
                    # np.savetxt(output_filename,
                    # np.vstack((np.arange(len(YpredTree)), YpredTree[:, ])).T,       # original: YpredTree[:, 1]
                    # '%d, %.2f', header='ID,Prob1', comments='', delimiter=',')

                else:
                    output_filename = 'results/Yhat_boosted_tree-' + str(n_training_data) \
                                      + '-nbags_' + str(nBags) \
                                      + '-nfeat_' + str(number_of_features_allowed) \
                                      + '-thrs_' + str(threshold) + '.txt'
                    # np.savetxt(output_filename,
                    # np.vstack((np.arange(len(YpredTree)), YpredTree[:, 1])).T,
                    # '%d, %.2f', header='ID,Prob1', comments='', delimiter=',')

                print "Saved : ", output_filename

if __name__ == '__main__':
    testRandomForest()
