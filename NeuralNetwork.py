# Neural Network Classifier
# Derives from Classifier in Learner.py
import numpy as np
from Utilities import toIndex, fromIndex, to1ofK, from1ofK
from numpy import asarray as arr
from numpy import atleast_2d as twod
from numpy import asmatrix as mat
from Learner import Classifier as Cl

class neuralNetworkClassifier(Cl.Classifier):
    __weights = arr([], dtype=object)

    __layers = []
    __dLayers = []

# init

# 

