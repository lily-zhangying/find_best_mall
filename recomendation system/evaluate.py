__author__ = 'John'
import numpy as np

def sort_prediction_all(predictions):
    #predictions finds the ranking of the as well as their corresponding indexes
    #assume predictions comes in a one-dimensional array
    sorted_values = np.sort(predictions);
    index = np.argsort(predictions); #this sorts by increasing order
    return(sorted_values[::-1], index[::-1])
    #return ( sorted_values, index[:, ::-1]) #this sorts by decreasing order

def ranked_precision(*kwargs):
    #kwargs = (predictions, truth, k)
    #does average precision for a person with 2 arguments
    #does precision for rank k with 3 arguments
    #finds the precision of the dataset at certain point k for a certain mall
    #note that the index for k starts at 1 rather the traditional 0. Because there is no 0th ranked item
    #obviously, truth an predictions must be the same
    predictions = kwargs[0]
    truth = kwargs[1]
    if(not (predictions.shape == truth.shape )):
        raise Exception("prediction and truth are not the same size")
    #sort predictions
    (sorted_pred, index) = sort_prediction_all(predictions)
    truth = truth[index];
    if(len(kwargs) ==2):
        each_precision = np.divide(np.cumsum(truth), np.arange(1, (truth.shape[0]+1)))
        average_precision = np.dot(each_precision, truth)/np.sum(truth)
        return(average_precision)
    if(len(kwargs) ==3):
        k = kwargs[2]
        return np.sum(truth[0:(k)])/k

#assume that test_indices are tuples
def map(X, X_predict, test_indices):
    #should be robust if it is for list and array
    #stable sorting on the indices
    index = np.argsort(test_indices[:, 0]);

    test_indices = test_indices[index, :] #probably wrong
    index = np.argsort(test_indices[:, 1]);
    test_indices = test_indices[index,:] #probably wrong
    #sorted_indices = sorted(test_indices,key=lambda x: x)
    n_users_in_test = 0
    val = 0
    # print(test_indices)
    # print(X_predict)
    for i in range(X.shape[1]):
        canidates = test_indices[ test_indices[:, 1] == i, :]
        if canidates.shape[0] >0:
            # print(i)
            # print(canidates[:, 0])
            #consider the case where u get all zeros
            #y_coordinates = i*np.ones(canidates.shape[0]).astype(int)
            prediction = X_predict[canidates[:, 0], canidates[:, 1]]
            truth = X[canidates[:, 0], canidates[:, 1]]
            #print(prediction)
            (sorted_pred, index) =sort_prediction_all(prediction)
            if(np.sum(truth) > 0):
                val = val + ranked_precision(prediction, truth)
                n_users_in_test = n_users_in_test +1
    return val/n_users_in_test;


#computes the rmse of the results
def negative_rmse(X, X_predict, test_indices):
    #print(X_predict[test_indices[0:10, 0], test_indices[0:10, 1]])
    return -1*np.sqrt( (np.average(np.square(X[test_indices[:, 0], test_indices[:, 1]] - X_predict[test_indices[:, 0], test_indices[:, 1]]))))


def rmse(X, X_predict, test_indices):
    #print(X_predict[test_indices[0:10, 0], test_indices[0:10, 1]])
    return np.sqrt( (np.average(np.square(X[test_indices[:, 0], test_indices[:, 1]] - X_predict[test_indices[:, 0], test_indices[:, 1]]))))


