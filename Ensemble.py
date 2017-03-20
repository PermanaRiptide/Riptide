# This is the Ensemble of learners (3+)
import numpy as np
import mltools as ml
import RandomForest as rf
import NeuralNetwork as nn

def thresholdsoft(preds):
    hardpreds = [None] * preds.shape[0]
    for i in range(preds.shape[0]):
        if preds[i] >= 0.5:
            hardpreds[i] = 1
        else:
            hardpreds[i] = 0
    return hardpreds

def classerror(hardpreds, actual):
    wrongs = 0
    for i in range(hardpreds.shape[0]):
        if hardpreds[i] != actual[i]:
            wrongs += 1
    return wrongs

np.random.seed(0)

# Xtr = np.genfromtxt("X_train.txt", delimiter=None)
# Ytr = np.genfromtxt("Y_train.txt", delimiter=None)
# ml.shuffleData(Xtr, Ytr)
# Xtr, Xva, Ytr, Yva = ml.splitData(Xtr, Ytr, 0.2)
#
# rF = rf.RandomForestClassifier(Xtr, Ytr, isboosted=False, num_learner=100, nFeatures=2)
# bF = rf.RandomForestClassifier(Xtr, Ytr, isboosted=True, num_learner=100, threshold=0.5, nFeatures=2, minLeaf=127, time_to_train=5)
# f = nn.NeuralNetworkFactory()
# nN, p = f.create(Xtr, Ytr)
#
#
# rftr = rF.err(Xtr, Ytr)
# rfva = rF.err(Xva, Yva)
#
# bftrp = bF.predict_soft(Xtr)
# bfvap = bF.predict_soft(Xva)
# bftr = classerror(thresholdsoft(bftrp), Ytr)
# bfva = classerror(thresholdsoft(bfvap), Yva)
#
# nntr = 1.0 - f.eval_model(Xtr, Ytr, p, nN, False)[0]
# nnva = 1.0 - f.eval_model(Xva, Yva, p, nN, False)[0]
#
# avgtr = (rftr + bftr + nntr) / 3.0
# avgva = (rfva + bfva + nnva) / 3.0
#
# print avgtr
# print avgva

Xtr = np.genfromtxt("X_train.txt", delimiter=None)
Ytr = np.genfromtxt("Y_train.txt", delimiter=None)
Xte = np.genfromtxt("X_test.txt", delimiter=None)
#
# # Train Models and Get Confidence Predictions
# print "Random Forest"
# randomForest = rf.RandomForestClassifier(Xtr, Ytr, isboosted=False, num_learner=100, nFeatures=2)
# rfpreds = randomForest.predict_soft(Xte)
#
# print "Boosted Forest"
# boostedForest = rf.RandomForestClassifier(Xtr, Ytr, isboosted=True, num_learner=100, threshold=0.5, nFeatures=2, minLeaf=127, time_to_train=5)
# bfpreds = boostedForest.get_confidence()
#
# print "Neural Network"
# factory = nn.NeuralNetworkFactory()
# neuralnet, p = factory.create(Xtr, Ytr)
# nnpreds = factory.predict(Xte, p, neuralnet)
# factory.kaggle(Xte, neuralnet, p, "NN_80N_5L_NoDrop_Actual.txt")

RF = np.genfromtxt("results/Yhat_boosted_tree_minLeaf64-100000-nbags_25-nfeat_2-thrs_0.5.txt", delimiter=",")[1:, 1]
BF = np.genfromtxt("inputdata/Yhat_dtree_esb200_nfeatures_2.txt", delimiter=",")[1:, 1]
NN = np.genfromtxt("NN_60N_5L_NoDrop.txt", delimiter=",")[1:, 1]
SCK1 = np.genfromtxt("Yhat_RF-weighted-nbags_100-AUC_0.65797.txt", delimiter=",")[1:, 1]
NN2 = np.genfromtxt("NN_50N_3L_NoDrop.txt", delimiter=",")[1:, 1]

AVG = RF + BF + NN + SCK1 + NN2
AVG /= 5.0
print AVG
np.savetxt('combine_4.txt', np.vstack( (np.arange(len(AVG)) , AVG) ).T, '%d, %.2f',header='ID,Prob1',comments='',delimiter=',')

# For Boosted Forest, since it is regression, push prediction values >= 0.5 as 1, else 0
# Then count the number of wrongs between the preds and actuals


'''
Boosted Forest: 64 Min Leaf, 25 Trees, 2 Features, 0.5 Threshold
Random Forest: 200 Trees, 2 Features
Neural Network: 80 Nodes, 5 Layers, No Dropout
Kaggle: 0.77070
'''
'''
Boosted Forest: 64 Min Leaf, 25 Trees, 2 Features, 0.5 Threshold
Random Forest: 200 Trees, 2 Features
Neural Network: 60 Nodes, 5 Layers, No Dropout
Sci Kit Forest: 100 Trees ~ Kaggle: 0.74354
Kaggle: 0.77177
'''
'''
Boosted Forest: 64 Min Leaf, 25 Trees, 2 Features, 0.5 Threshold
Random Forest: 200 Trees, 2 Features
Neural Network: 60 Nodes, 5 Layers, No Dropout
Sci Kit Forest: 100 Trees ~ Kaggle: 0.74354
Neural Network 2: 50 Nodes, 3 Layers, No Dropout
Kaggle: 0.77186
'''
