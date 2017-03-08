
def testNN():
    # Imports
    import NeuralNetwork as nn
    import numpy as np
    import matplotlib.pyplot as plt
    import Utilities as util

    np.random.seed(0)  # Set seed to get deterministic results

    # Get data from text
    print "GENERATING DATA FROM TEXT"
    x = np.genfromtxt("X_train.txt", delimiter="", max_rows=10000)
    y = np.genfromtxt("Y_train.txt", delimiter="", max_rows=10000)

    print "SHUFFLING AND SPLITTING DATA"
    x, y = util.shuffleData(x, y)  # Shuffle data
    xtr, xva, ytr, yva = util.splitData(x, y, 0.75)  # Split to test and validation sets


    print "TRAINING NEURAL NET"
    c = [0, 1]
    s = [14, 14, 2]; i = 'zeros'; ss = 0.1; t = 1e-4; ms = 5000  # nnet training parameters

    nl = 5; nln = 14; s = [14]
    for n in range(nl):
        s.append(nln)

    print s

    etr = [None] * nl; eva = [None] * nl
    for l in range(1, nl + 1):
        nnet = nn.NeuralNetworkClassifier(c)  # Train a nnet model
        st = s[:l + 1]
        st.append(2)
        print "Layers: ", st
        nnet.init_weights(st, i)
        nnet.train(xtr, ytr, ss, t, ms)
        etr[l - 1] = nnet.err(xtr, ytr)
        eva[l - 1] = nnet.err(xva, yva)

    plt.plot(range(nl), etr, "y", label="Training")
    plt.plot(range(nl), eva, "g", label="Validation")
    plt.legend()
    plt.show()

    # print nnet.num_classes()

    # etr = nnet.err(xtr, ytr)
    # eva = nnet.err(xva, yva)
    # atr = nnet.auc(xtr, ytr)
    # ava = nnet.auc(xva, yva)

    print "Training Error: ", etr
    print "Validation Error: ", eva
    # print "Training AUC: ", atr
    # print "Validation AUC: ", ava


def testJupyter():
    import numpy as np
    np.random.seed(0)
    import mltools as ml
    import matplotlib.pyplot as plt   # use matplotlib for plotting with inline plots
    import mltools.nnet    # import neural network code

    iris = np.genfromtxt("data/iris.txt",delimiter=None)
    # x = np.genfromtxt("X_train.txt", delimiter="", max_rows=10000)
    # y = np.genfromtxt("Y_train.txt", delimiter="", max_rows=10000)
    X, Y = iris[:,0:2], iris[:,-1]   # get first two features & target
    X,Y  = ml.shuffleData(X,Y)       # reorder randomly (important later)
    X,_  = ml.transforms.rescale(X)  # works much better on rescaled data

    XA, YA = X[Y<2,:], Y[Y<2]        # get class 0 vs 1
    XB, YB = X[Y>0,:], Y[Y>0]        # get class 1 vs 2

    nn = ml.nnet.nnetClassify()
    nn.init_weights( [2,2], 'random', XA,YA)

    nn.train(XA, YA, stopTol=1e-8, stepConstant=.25, stopIter=300)
    ml.plotClassify2D(nn,XA,YA)
    print "\n",nn.wts
    plt.show()
    nn = ml.nnet.nnetClassify()
    nn.init_weights( [2,5,3], 'random', X,Y)

    nn.train(X, Y, stopTol=1e-8, stepConstant=.3, stopIter=2000)
    ml.plotClassify2D(nn,X,Y)
    plt.show()

##################
# Run tests here #
##################
# testNN()

testJupyter()