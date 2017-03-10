
def testRandomForest():
    # Imports
    import RandomForest as rf
    import numpy as np
    import matplotlib.pyplot as plt
    import Utilities as util

    np.random.seed(0)  # Set seed to get deterministic results

    # Get data from text
    print "GENERATING DATA FROM TEXT"
    x = np.genfromtxt("X_train.txt", delimiter="", max_rows=100000)
    y = np.genfromtxt("Y_train.txt", delimiter="", max_rows=100000)

    # print y
    # print np.unique(y)

    print "SHUFFLING AND SPLITTING DATA"
    # x, y = util.shuffleData(x, y)  # Shuffle data
    xtr, xva, ytr, yva = util.splitData(x, y, 0.9)  # Split to test and validation sets

    print "TRAINING RANDOM FOREST"

    my_threshold = [0.5]
    my_allowed_features = [3]


    for threshold in my_threshold:
        for number_of_features_allowed in my_allowed_features:

            # Train a random forest model :
            my_random_forest = None
            my_random_forest = rf.RandomForestClassifier(xtr, ytr, 25, 0.5, nFeatures=number_of_features_allowed)

            # print "my_threshold               = ", threshold
            # print "number_of_features_allowed = ", number_of_features_allowed

            # etr = my_random_forest.err(xtr, ytr)
            # print "Training Error: ", etr

            # eva = my_random_forest.err(xva, yva)
            # print "Validation Error: ", eva

            # atr = my_random_forest.auc(xtr, ytr)
            # print "Training AUC: ", atr

            ava = my_random_forest.auc(xva, yva)
            print "Validation AUC = ", ava
            print "------------------------------------------------------"

##################
# Run tests here #
##################
testRandomForest()