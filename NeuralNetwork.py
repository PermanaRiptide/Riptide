# Script for training the Neural Network, and finding the best parameters
import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def what_do_I_import():
    print "Required import statements"
    print "import numpy as np"
    print "import mltools as ml"
    print "from keras.models import Sequential"
    print "from keras.layers import Dense"
    print "from sklearn.metrics import roc_auc_score"
    print "In case of changes:"
    print "from keras.layers import Dropout"
    print "from sklearn.model_selection import StratifiedKFold"
    print "import matplotlib.pyplot as plt"

class NeuralNetworkFactory:
    __first_time = False

    def __init__(self, debug=False):
        # print "Neural Network Factory"
        if NeuralNetworkFactory.__first_time:
            what_do_I_import()
        self.__debug = debug
        NeuralNetworkFactory.__first_time = False

    def create(self, x, y):
        if self.__debug:
            print "Creating Model..."
        m, p = self.__create_model(x, y)
        return m, p

    def eval_model(self, x, y, p, m, pr):
        if self.__debug:
            print "Evaluating Model with Batch Size 15"
        x = self.__transformx(x, p)
        scores = m.evaluate(x, y, batch_size=15, verbose=0)
        pred = m.predict_proba(x, batch_size=15, verbose=0)
        auc = roc_auc_score(y, pred)
        if pr or self.__debug:
            print "Training Accuracy: ", scores[1]
            print "Training AUC: ", auc
        return scores[1], auc

    def predict(self, x, p, m):
        x = self.__transformx(x, p)
        return m.predict_proba(x)

    def kaggle(self, x, m, p, text_file):
        x = self.__transformx(x, p)
        pred = m.predict_proba(x)
        if self.__debug:
            print "Saving to File: ", text_file
        np.savetxt(text_file, np.vstack((np.arange(len(pred)), pred.T[0])).T, '%d, %.2f', header='ID,Prob1', comments="", delimiter=",")

    def __transformxy(self, x, y):
        if self.__debug:
            print "Transforming X to degree 2"
        x = ml.transforms.fpoly(x, 2, bias=False)
        x, p = ml.transforms.rescale(x)
        return x, y, p

    def __transformx(self, x, p):
        if self.__debug:
            print "Transforming Test X to degree 2"
        x = ml.transforms.fpoly(x, 2, bias=False)
        x, _ = ml.transforms.rescale(x, p)
        return x

    def __create_model(self, x, y):
        x, y, p = self.__transformxy(x, y)
        m = Sequential()
        if self.__debug:
            print "Adding Layers (5) with 60 Nodes each to Model"
        m.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform', activation='relu'))
        m.add(Dense(50, kernel_initializer='uniform', activation='relu'))
        m.add(Dense(50, kernel_initializer='uniform', activation='relu'))
        # m.add(Dense(80, kernel_initializer='uniform', activation='relu'))
        # m.add(Dense(80, kernel_initializer='uniform', activation='relu'))
        m.add(Dense(1,  kernel_initializer='uniform', activation='sigmoid'))
        if self.__debug:
            print "Model Summary: "
            print m.summary()

        if self.__debug:
            print "Compiling Model with Binary Cross Entropy loss, Adam Optimizier"
        m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        if self.__debug:
            print "Training Model with 150 Epochs, 15 Batch Size"
        m.fit(x, y, epochs=150, batch_size=15, verbose=2)
        return m, p


"""
        x, y, p = self.__transformxy(x, y)
        m = Sequential()
        if self.__debug:
            print "Adding Layers (5) with 60 Nodes each to Model"
        m.add(Dense(60, input_dim=x.shape[1], kernel_initializer='uniform', activation='relu'))
        m.add(Dense(60, kernel_initializer='uniform', activation='relu'))
        m.add(Dense(60, kernel_initializer='uniform', activation='relu'))
        m.add(Dense(60, kernel_initializer='uniform', activation='relu'))
        m.add(Dense(60, kernel_initializer='uniform', activation='relu'))
        m.add(Dense(1,  kernel_initializer='uniform', activation='sigmoid'))
        if self.__debug:
            print "Model Summary: "
            print m.summary()
"""
''' Example Run
seed = 27
np.random.seed(seed)

X = np.genfromtxt("X_train.txt", delimiter="")
Y = np.genfromtxt("Y_train.txt", delimiter="")
Xte = np.genfromtxt("X_test.txt", delimiter="")

NN = NeuralNetworkFactory(True)
model, p = NN.create(X, Y, Xte)
NN.eval_model(X, Y, p, model, True)
NN.kaggle(Xte, model, p, "nny_test.txt")
'''

# Scores for Reference
'''
-14 F, 30 N, 1 L, 200 E, 30 B, 0.2 S, 2 D, Adam O, Relu/Sig A, Uniform I
    TAC: 71.1270001993
    TAU: 0.716657749278
    Kaggle: 0.70610
-14 F, 80 N, 6 L, 150 E, 10 B, 0.2 S, 2 D, Adam O, Relu/Sig A, Uniform I
    Training Accuracy Score:  0.733112500776
    Training AUC Score:  0.751246538346
    Test Accuracy Score:  0.713150002107
    Test AUC Score:  0.706857903065
    Kaggle:

2 L with 60 Nodes each and Dropout 0.2 has:
Training Accuracy Score:  0.716637501784
Training AUC Score:  0.72649009824
Test Accuracy Score:  0.708100002393
Test AUC Score:  0.706397900298

3 L with 80 Nodes each and Dropout 0.2 has:
Training Accuracy Score:  0.713125002086
Training AUC Score:  0.727842431236
Test Accuracy Score:  0.703450002566
Test AUC Score:  0.708697847444

5 L with 80 Nodes each and Dropout 0.2 has:
Training Accuracy Score:  0.709837502342
Training AUC Score:  0.717078415717
Test Accuracy Score:  0.705550002262
Test AUC Score:  0.702505235322

5 L with 60 Nodes each and Dropout 0.2 has:
Training Accuracy Score:  0.701850003242
Training AUC Score:  0.70147817481
Test Accuracy Score:  0.702950002804
Test AUC Score:  0.694798559268

5 L with 60 Nodes each and Dropout 0.5 has:
Training Accuracy Score:  0.69626250346
Training AUC Score:  0.645492350208
Test Accuracy Score:  0.697800004259
Test AUC Score:  0.645098064845


Try with 5 L with 60 Nodes each and no Dropout. Train on ALL data
Training Accuracy Score:  0.740840000641
Training AUC Score:  0.776536089158
Kaggle: 0.72767

3 L with 60 N each, no Dropout. All data
Training Accuracy Score:  0.737600000894
Training AUC Score:  0.768473991757
Kaggle: 0.72428

1 L with 60 N each, no Dropout. ALL data
Training Accuracy Score:  0.716400001374
Training AUC Score:  0.72671486012

5L with 100 N each, no Dropout. All data
Training Accuracy Score:  0.755299999723
Training AUC Score:  0.797745298572
Kaggle: 0.72478

5L with 100 N each, Dropout 0.2. All data
Training Accuracy Score:  0.712830002227
Training AUC Score:  0.720305294854
Kaggle: 0.71230
'''

# Best AUC Score is 5 Layers with 60 Nodes each with no Dropout
# Add more layers and reduce Nodes?
# Try 2 Layers with 60 Nodes each
# Or 4 Layers with 60 Nodes each
# Or 3 Layers with 80 Nodes each



#
# def do_neural(x, y, nodes, layers, feats, batch, epochs, drop):
#     Model = Sequential()
#     Model.add(Dense(nodes, input_dim=feats, kernel_initializer=init, activation=act))
#     # Model.add(Dropout(drop))
#     for i in range(layers - 1):
#         Model.add(Dense(nodes, kernel_initializer=init, activation=act))
#         # Model.add(Dropout(drop))
#     Model.add(Dense(output_nodes, kernel_initializer=init, activation=output_act))
#     Model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
#     Model.fit(x, y, epochs=epochs, batch_size=batch, verbose=verbose)
#     return Model
#
# def evalsa(x, y, Model):
#     scores = Model.evaluate(x, y, batch_size=batch, verbose=verbose)
#     p = Model.predict_proba(x, batch_size=batch, verbose=verbose)
#     auc = roc_auc_score(y, p)
#     return scores[1], auc
#
#
# def kaggle(model, xte):
#     p = model.predict_proba(xte)
#     np.savetxt("nny_100x5_dr20.txt", np.vstack((np.arange(len(p)), p.T[0])).T, '%d, %.2f', header='ID,Prob1', comments="", delimiter=",")
#
#
# def transform(x, y, d, f, p=None):
#     x = ml.transforms.fpoly(x, d, bias=False) if degree > 0 else x
#     if p is None:
#         x, p = ml.transforms.rescale(x) if d > 0 else (x, 0)
#     else:
#         x, p = ml.transforms.rescale(x, p) if d > 0 else (x, 0)
#     new_f = x.shape[1] - f if d > 0 else 0
#     return x, y, new_f, p
#
#
# def graph(xlist, ylist, clist, llist):
#     for x, y, c, l in zip(xlist, ylist, clist, llist):
#         plt.plot(x, y, color=c, label=l)
#         plt.legend()
#     plt.show()


# Get Data

# # Training and Validation Split (w/o CV)
# Xt = X[:80000, :]
# Yt = Y[:80000]
# Xv = X[80000:, :]
# Yv = Y[80000:]
#
# # TRIGGERS #
# do_kfolds = False
# do_val = False
# try_fpoly = True
# do_final = True
# custom = False
# # Num of Features: 14
# # Degrees: 2
# # try_features = [1, 2, 4, 6, 8, 10, 12, 14]
# # try_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# # try_list = [1, 2, 3, 4, 5, 6, 7, 8]
# # try_list = [50, 100, 150, 200, 250, 300]
# try_list = [1]
#
# nf = 0
# Xp, Yp = X, Y
# if try_fpoly:
#     if do_kfolds:
#         Xp, Yp, nf, p = transform(X, Y, degree, num_features)
#         Xte, _, nf, _ = transform(Xte, Y, degree, num_features, p)
#     if do_val:
#         Xt, Yt, nf, p = transform(Xt, Yt, degree, num_features)
#         Xv, Yv, nf, _ = transform(Xv, Yv, degree, num_features, p)
#         Xte, _, nf, _ = transform(Xte, Y, degree, num_features, p)
#     if do_final:
#         X, Y, nf, p = transform(X, Y, degree, num_features)
#         Xte, _, nf, _ = transform(Xte, Y, degree, num_features, p)
#
# # slist = []
# # alist = []
# # splist = []
# # aplist = []
# strl = []
# stel = []
# atrl = []
# atel = []
#
# # Set up KFolds
# if do_kfolds:
#     kfold = StratifiedKFold(n_splits=ksplits, shuffle=True, random_state=seed)
#     for t in try_list:
#         kstr = 0
#         katr = 0
#         kste = 0
#         kate = 0
#         k = 1
#         for train, test in kfold.split(Xp[:, :14 + nf], Yp):
#             print t, ", ", k
#             k += 1
#             Model = do_neural(Xp[train], Yp[train], nodes=20, layers=t, feats=14 + nf, batch=100, epochs=50)
#             sr, ar = evalsa(Xp[train], Yp[train], Model)
#             se, ae = evalsa(Xp[test], Yp[test], Model)
#             print ""
#             print "Training Accuracy Score: ", sr
#             print "Training AUC Score: ", ar
#             print "Test Accuracy Score: ", se
#             print "Test AUC Score: ", ae
#             kstr += sr
#             katr += ar
#             kste += se
#             kate += ae
#
#         kstr /= ksplits
#         katr /= ksplits
#         kste /= ksplits
#         kate /= ksplits
#         print ""
#         print "KFold Training Accuracy Score: ", kstr
#         print "KFold Training AUC Score: ", katr
#         print "KFold Test Accuracy Score: ", kste
#         print "KFold Test AUC Score: ", kate
#         strl.append(kstr)
#         atrl.append(katr)
#         stel.append(kste)
#         atel.append(kate)
#     x = [try_list, try_list, try_list, try_list]
#     y = [strl, atrl, stel, atel]
#     c = ['m', 'c', 'y', 'g']
#     l = ['Tr Accuracy', 'Tr AUC', 'Te Accuracy', 'Te AUC']
#     graph(x, y, c, l)
# if do_val:
#     # Best Total Nodes with Batch = 100, Epoch = 100
#     # try_list = [80, 100, 120, 140, 160]
#     # Best Total Layers with 120 Fixed Nodes
#     # Try 2 Layers, 60 each
#     try_list = [5, 6, 7, 8, 9, 10]
#     for t in try_list:
#         print "########### ", t
#         Model = do_neural(Xt, Yt, nodes=100, layers=t, feats=14 + nf, batch=100, epochs=100)
#         print "Model Statistics: "
#         print Model.summary()
#         sr, ar = evalsa(Xt, Yt, Model)
#         se, ae = evalsa(Xv, Yv, Model)
#         print ""
#         print "Training Accuracy Score: ", sr
#         print "Training AUC Score: ", ar
#         print "Test Accuracy Score: ", se
#         print "Test AUC Score: ", ae
#         strl.append(sr)
#         atrl.append(ar)
#         stel.append(se)
#         atel.append(ae)
#
#     x = [try_list, try_list, try_list, try_list]
#     y = [strl, atrl, stel, atel]
#     c = ['m', 'c', 'y', 'g']
#     l = ['Tr Accuracy', 'Tr AUC', 'Te Accuracy', 'Te AUC']
#     graph(x, y, c, l)
# if custom:
#     Model = do_neural(Xt, Yt, nodes=100, layers=7, feats=14 + nf, batch=15, epochs=150)
#     print "Model Statistics: "
#     print Model.summary()
#     sr, ar = evalsa(Xt, Yt, Model)
#     se, ae = evalsa(Xv, Yv, Model)
#     print ""
#     print "Training Accuracy Score: ", sr
#     print "Training AUC Score: ", ar
#     print "Test Accuracy Score: ", se
#     print "Test AUC Score: ", ae
#
# if do_final:
#     Model = do_neural(X, Y, nodes=100, layers=5, feats=14 + nf, batch=20, epochs=150, drop=0.2)
#     # sr, ar = evalsa(Xt, Yt, Model)
#     # se, ae = evalsa(Xv, Yv, Model)
#     sr, ar = evalsa(X, Y, Model)
#     print ""
#     print "Training Accuracy Score: ", sr
#     print "Training AUC Score: ", ar
#     # print "Test Accuracy Score: ", se
#     # print "Test AUC Score: ", ae
#     kaggle(Model, Xte)

'''
Total Nodes: 120 best, but peak is at 140
Total Layers with Fixed Total Nodes: 2 seems to do the best, but 5 may be decent as well (it may be that there aren't enough nodes @ ~0.75 Training, ~0.71 Test
Total Layers with Constant Nodes:

'''



# Store Scores
# acclist = []
# auclist = []
# Loop for Best # of Nodes
# for i in range(1, num_nodes + 1):
# Create Model
# print "Training on HN = ", i

# Model = Sequential()
# Model.add(Dense(num_nodes, input_dim=num_features + new_f, kernel_initializer=init, activation=act))
# Model.add(Dense(output_nodes, kernel_initializer=init, activation=output_act))
#
# Model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
#
# # Train Model
# Model.fit(Xtr, Ytr, epochs=epochs, batch_size=30, validation_split=split, verbose=verbose)
#
# # Evaluate Model
# scores = Model.evaluate(Xtr, Ytr, batch, verbose=verbose)
# print "Training Accuracy: ", scores[1] * 100, "%"
# acclist.append(scores[1])
# p = Model.predict_proba(Xtr, batch_size=batch, verbose=verbose)
# auc = roc_auc_score(Ytr, p)
# auclist.append(auc)
# print "Training AUC: ", auc



# Graph
# x = range(1, num_nodes + 1)
# plt.plot(x, acclist, "y", x, auclist, "g")
# plt.show()



