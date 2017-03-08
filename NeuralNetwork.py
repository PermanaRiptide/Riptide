# Neural Network Classifier
# Derives from Classifier in Learner.py
import numpy as np
import Utilities as util
import Learner as L
from numpy import asarray as arr
from numpy import asmatrix as mat
from numpy import atleast_2d as twod
from numpy import concatenate as concat
from numpy import column_stack as cols


class NeuralNetworkClassifier(L.Classifier):
    """
    Neural Network Classifier derived from Classifier base.
    """
    # Constructor for the Neural Network Classifier
    # This will train the classifier automatically, if the data is passed in
    def __init__(self, *args, **kwargs):
        # M x N array of weights, where:
        # M: The number of weights per layer
        # N: The number of layers
        self.__classes = []
        self.__weights = arr([], dtype=object)
        # Functions: one for the Hidden Layers, and one for the Activation Layer
        # Each function has an accompanying derivative function
        # Recommendations: tanh(x) for Hidden Function, and sigmoid(x) for Activation
        self.__actHidden = lambda z: np.tanh(z)
        self.__dActHidden = lambda z: np.tanh(z)**2
        self.__actOut = lambda z: np.tanh(z)
        self.__dActOut = lambda z: np.tanh(z)**2
        if len(args) or len(kwargs):
            self.train(*args, **kwargs)

    # Represents the Classifier
    def __repr__(self):
        return str(self)

    # Turns the Classifier into a string
    def __str__(self):
        return "Neural Net: [{}]".format(self.get_layers())

    # Returns the number of layers (which is the number of columns)
    def num_layers(self):
        return len(self.__weights)

    # Returns the layers of the Neural Network as a list
    # If there are no layers, returns an empty list
    @property
    def layers(self):
        if len(self.__weights):
            layers = [self.__weights[la].shape[1] for la in range(len(self.__weights))]
            layers.append(self.__weights[-1].shape[0])
        else:
            layers = []
        return layers

    def train(self, x, y, sizes=[], init='zeros', step_size=0.1, tolerance=1e-4, max_steps=5000):
        self.init_weights(sizes, init)
        if self.__weights[0].shape[1] - 1 != len(x[0]):
            raise ValueError("First Layer must equal the number of columns of x (number of features)")

        self.__classes = self.__classes if len(self.__classes) else np.unique(y)

        print len(self.__classes)
        print self.__weights[-1].shape[0]

        if len(self.__classes) != self.__weights[-1].shape[0]:
            raise ValueError("Final Layer must equal the number of classes in y")



        y_tr_k = util.to1ofK(y)

        m, n = mat(x).shape
        layers = len(self.__weights)

        num_iter = 1
        done = False
        j_cl, j_sur = [], []

        while not done:
            step_i = step_size / num_iter

            for i in range(n):
                a, z = self.__responses(x[i, :])
                delta = (z[layers] - y_tr_k[i, :]) * arr(self.__dActOut(z[layers]))

                for l in range(layers - 1, -1, -1):
                    gradient = delta.T.dot(z[l])
                    delta = delta.dot(self.__weights[l]) * self.__dActHidden(z[l])
                    delta = delta[:, 1:]
                    self.__weights[l] -= step_i * gradient

            j_cl.append(self.err_k(x, y_tr_k))
            j_sur.append(self.mse_k(x, y_tr_k))

            done = (num_iter > 1) and (np.abs(j_sur[-1] - j_sur[-2]) < tolerance) or num_iter >= max_steps
            num_iter += 1

    def predict(self, x):
        raise NotImplementedError()

    def predict_soft(self, x):
        """
        Make a soft prediction (per class confidence) of the neural network on x
        :param x: M x N matrix of M data points with N features each
        :return: M x 1 matrix of M predicted classes per data point
        """
        x = arr (x)  # Converts to matrix
        layers = self.num_layers()  # Number of layers
        z = self.__addOne(x)  # Input features + Constant Term

        for l in range(layers - 2):  # All layers but the output layer
            z = self.__linearResponse(z, l)  # Compute linear reponse for Next Layer
            z = self.__addOne(self.__actHidden(z))  # Activation for Hidden Layers

        z = self.__linearResponse(z, layers - 1)  # Compute linear response for Output

        return self.__actOut(z)  # Activation for Output Layer

    def err_k(self, x, y):
        y_hat = self.predict(x)
        return np.mean(y_hat != util.from1ofK(y))

    def log_likelihood(self, x, y):
        r, c = twod(y).shape
        if r == 1 and c != 1:
            y = twod(y).T

        soft = self.predict_soft(x)
        return np.mean(np.sum(np.log(np.power(soft, y, )), 1), 0)

    def mse(self, x, y):
        return self.mse_k(x, util.to1ofK(y))

    def mse_k(self, x, y):
        return np.power(y - self.predict_soft(x), 2).sum(1).mean(0)

    def set_activation(self, method, hidden=None, out=None, dHidden=None, dOut=None):
        method.method.lower()
        if hidden is not None and dHidden is None or \
            hidden is None and dHidden is not None or \
            out is not None and dOut is None or \
            out is None and dOut is not None:
            raise ValueError("Cannot have an Activation Function without its derivative, and vice-versa")

        if method == 'logistic':
            self.__actHidden = lambda z: twod(1.0 / 1.0 + np.exp(-z))
            self.__dActHidden = lambda z: twod(np.multiply(self.__actHidden(z), (1 - self.__actHidden(z))))
            self.__actOut = self.__actHidden
            self.__dActOut = self.__dActHidden
        elif method == 'htangent':
            self.__actHidden = lambda z: twod(np.tanh(z))
            self.__dActHidden = lambda z: twod(1 - np.power(np.tanh(z)), 2)
            self.__actOut = self.__actHidden
            self.__dActOut = self.__dActHidden
        elif method == 'custom':
            self.__actHidden = hidden
            self.__dActHidden = dHidden
            self.__actOut = out
            self.__dActOut = dOut
        else:
            raise ValueError("method type not recognized: " + str(method))

    def set_layers(self, sizes, init='random'):
        self.init_weights(sizes, init)

    def init_weights(self, sizes, init):
        init = init.lower()

        if init == 'none':
            pass
        elif init == 'zeros':
            self.__weights = arr([np.zeros((sizes[i + 1], sizes[0] + 1)) for i in range(len(sizes) - 1)], dtype=object)
        elif init == 'random':
            self.__weights = arr([0.0025 *
                                  np.random.randn(sizes[i + 1], sizes[i] + 1)
                                  for i in range(len(sizes) - 1)], dtype=object)
        else:
            raise ValueError("init type not recognized: " + str(init))

    def get_layers(self):
        s = arr([mat(self.__weights[i]).shape[1] - 1 for i in range(len(self.__weights))])
        s = concat((s, [mat(self.__weights[-1]).shape[0]]))
        return s

    def __addOne(self, x):
        return np.hstack((np.ones((x.shape[0], 1)), x))

    def __linearResponse(self, z, wgt):
        return mat(z) * mat(self.__weights[wgt]).T

    def __responses(self, wgts, x_in, hidden, out):
        layers = len(wgts)
        const_feat = np.ones((max(x_in).shape[0], 1)).flatten()
        a = [arr([1])]
        z = [concat((const_feat, x_in))]

        for l in range(1, layers):
            a.append(z[l - 1].dot(wgts[l - 1].T))
            z.append(cols((np.ones((mat(a[l]).shape[0], 1)), hidden(a[l]))))

        a.append(arr(mat(z[layers - 1]) * mat(wgts[layers - 1]).T))
        z.append(arr(out(a[layers])))
        return a, z

# init

# train

# predict

# predictSoft

# (stochastic) gradient descent

# back propagation

# forward propagation

# loss

