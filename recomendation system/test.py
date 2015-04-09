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




#ranked_precision()
prediction = np.arange(20, 0, -1)

X_predict = np.array([prediction.T, prediction.T])
X_predict = X_predict.T
X =np.array([[1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0] ])
X=X.T
ind = np.random.permutation(20)
X_predict = X_predict[ind]
X = X[ind]
#the prediction and the ground truth are randomly permuted
selected = np.arange(20)[ X_predict[:, 0] > 10]
ones = np.array([selected.T, np.ones(10).T.astype(int)]).T
zeros = np.array([selected.T, np.zeros(10).T.astype(int)]).T
test_ind = np.row_stack((ones, zeros))
print(selected)
# (sorted_pred, index) = sort_prediction_all(X_predict[ones[:, 0], ones[:, 1]])
# print(X[index, 1])


print(map(X, X_predict, test_ind))

# prediction = prediction[::-1]
# ind = np.random.permutation(10)
# truth = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1])
# # print(prediction[ind])
# # print([truth[ind]])
#
# print(ranked_precision(prediction, truth))