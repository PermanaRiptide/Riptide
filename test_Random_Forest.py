
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

    my_random_forest = rf.RandomForestClassifier(xtr, ytr, 25, 0.5, nFeatures=2)  # Train a random forest model

    # etr = my_random_forest.err(xtr, ytr)
    eva = my_random_forest.err(xva, yva)
    # atr = my_random_forest.auc(xtr, ytr)
    ava = my_random_forest.auc(xva, yva)

    # print "Training Error: ", etr
    print "Validation Error: ", eva

    # print "Training AUC: ", atr
    print "Validation AUC: ", ava


##################
# Run tests here #
##################
testRandomForest()