__author__ = 'John'

X_predict = np.array([prediction.T, prediction.T])
X_predict = X_predict.T
X =np.array([[1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0] ])
X=X.T
# ind = np.random.permutation(20)
# X_predict = X_predict[ind]
# X = X[ind]
# #the prediction and the ground truth are randomly permuted
# selected = prediction[ X_predict[:, 0] < 10]
# ones = np.array([selected.T, np.ones(10).T.astype(int)]).T
# zeros = np.array([selected.T, np.zeros(10).T.astype(int)]).T
# test_ind = np.row_stack((ones, zeros))
# (sorted_pred, index) = sort_prediction_all(X_predict[ones[:, 0], ones[:, 1]])
#print(X[index, 1])
test_ind = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0 ,0, 0,  0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
test_ind= test_ind.T
print(map(X, X_predict, test_ind))
