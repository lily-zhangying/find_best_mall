import numpy as np
from numpy.linalg import norm

def test(name):
    print(name)
def hello(a, b):
    print(a, b)

dummy = list()
dummy.append("a")
dummy.append("b")
dummy.append("c")
l = [1,2,3,7]
print("(" + ", ".join([str(x) for x in l] ) +")")