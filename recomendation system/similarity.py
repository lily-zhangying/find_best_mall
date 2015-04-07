__author__ = 'John'
import numpy as np
import math
from numpy.linalg import norm
#all of the functions return the lambda function

def cosine():
    return lambda x, y: np.dot(x, y)/ (norm(x)*norm(y))

def gaussian(alpha=1, sigma=1):
    return lambda x, y: alpha* np.exp(-1*pow(norm(x-y), 2)/sigma)