
def testNN():
    # Imports
    import NeuralNetwork as nn
    import numpy as np
    import matplotlib.pyplot as plt
    import Utilities as util

    np.random.seed(0)  # Set seed to get deterministic results

    # Get data from text
    print "GENERATING DATA FROM TEXT"
    x = np.genfromtxt("X_train.txt", delimiter="", max_rows=1000)
    y = np.genfromtxt("Y_train.txt", delimiter="", max_rows=1000)

    print y
    print np.unique(y)

    print "SHUFFLING AND SPLITTING DATA"
    x, y = util.shuffleData(x, y)  # Shuffle data
    xtr, ytr, xva, yva = util.splitData(x, y, 0.75)  # Split to test and validation sets

    print "TRAINING NEURAL NET"
    s = [14, 14, 2]; i = 'zeros'; ss = 0.1; t = 1e-4; ms = 5000  # nnet training parameters
    nnet = nn.NeuralNetworkClassifier(xtr, ytr, s, i, ss, t, ms)  # Train a nnet model

    etr = nnet.err(xtr, ytr)
    eva = nnet.err(xva, yva)
    atr = nnet.auc(xtr, ytr)
    ava = nnet.auc(xva, yva)

    print "Training Error: ", etr
    print "Validation Error: ", eva
    print "Training AUC: ", atr
    print "Validation AUC: ", ava


##################
# Run tests here #
##################
testNN()