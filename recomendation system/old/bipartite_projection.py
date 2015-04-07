__author__ = 'John'
import numpy as np;
import random
import math
import scipy.sparse.bsr
from sklearn.cross_validation import train_test_split, KFold
from numpy.linalg import inv
from sklearn.decomposition import ProjectedGradientNMF
from itertools import groupby
np.random.seed(42);

class one_class:
    def __init__(self, data =None, item_mat = None, mall_mat = None):
        X = data; #X is the matrix that we are dealing with. The rows are items and columns are users.
        #training_data #binary mat determining which is enteries are in the training set
        #testing_data #binary mat determining which is enteries are in the testing set
        test = {0:None, 1:None, "data":None}
        train = {0:None, 1:None, "data":None}
        self.mall_mat = mall_mat
        self.item_mat = item_mat
            #default constants
        #cons =




    def wlas(self, R, W, d, V_init=None, cons = None):
        #if c is empty not assigned a value, it should be found through finding a constant
        #V should be found through nonnegative matri factorizations

        #May need more testing
        #initializing stuff here
        #assume R, W, V_init are matricies
        #now i need to parameter searching with cross validation
        #http://scikit-learn.org/stable/modules/grid_search.html

            # grid = GridSearchCV(KernelDensity(kernel='gaussian'),
            #         {'bandwidth': np.linspace(0.1, 1.0, 30)}
            # use cv method and predict for all the on the test set 1s

        V = V_init
        Nrow, Ncol = R.shape
        U = np.zeros((Nrow, d))
        I = np.eye(d)
        for k in range(cons["iter_max"]):
            for i in range(Nrow):
                W_hat_i = np.diag(W[i, :])
                (V.T).dot(W_hat_i)
                psm = (V.T).dot(W_hat_i).dot(V) +cons["lambda"]*sum(W[i, :])*I #the long matrix in the paper that is claimed to be semi positive definite
                U[i, :] = R[i, :].dot(W_hat_i).dot(V).dot(inv(psm))
            for j in range(Ncol):
                W_hat_j = np.diag(W[:, j])
                psm = U.T.dot(W_hat_j).dot( U) + cons["lambda"]*sum(W[:, j])*I
                V[j, :] = (R[:, j].T).dot(W_hat_j).dot(U).dot(inv(psm))
            #computing error
            diff = R - np.dot(U, V.T)
            diff_square = np.multiply(diff, diff)
            error = sum(sum(np.multiply(W, diff_square)))
            print(error)
            if(error < cons["error"]):
                break


        return (W, V)




    #partitions data into training and testing data by percentage
    def cv_percent(self, percent):
        #keywords:
        #percent - the percent that you want the traiining data to be random
        #folds - number of folds you are working with

        #find indices of ones and put them into training/testing sets
        ones_x, ones_y = np.nonzero(self.X[: ,:] == 1)
        one_coord =np.array([ones_x, ones_y]);
        one_coord = one_coord.T
        ones_train, ones_test = train_test_split(one_coord, test_size=percent, random_state=42)
        self.train[1] = ones_train
        self.test[1] = ones_test

        #find indices of ones and put them into training/testing sets

        zero_x, zero_y = np.nonzero(self.X[: ,:] == 0)
        zero_coord = np.array([zero_x, zero_y]);
        zero_coord = zero_coord.T
        zero_train, zero_test = train_test_split(zero_coord, test_size=percent, random_state=42)
        self.train[0] = zero_train
        self.test[0] = zero_test

        self.train["data"]= np.concatenate((ones_train, zero_train),axis=0)
        self.test["data"]= np.concatenate((ones_test, zero_test),axis=0)
        #concatenate the training and test array

    #not done
    def kfold(self, k):
        #divide dataset into chucks of sets
        ones_x, ones_y = np.nonzero(self.X[: ,:] == 1)
        one_coord =np.array([ones_x, ones_y]);
        one_coord = one_coord.T

        zero_x, zero_y = np.nonzero(self.X[: ,:] == 0)
        zero_coord = np.array([zero_x, zero_y]);
        zero_coord = zero_coord.T
        N_ones = one_coord.shape[0];
        N_zeros = zero_coord.shape[0];
        one_kfold = KFold(n=N_ones, n_folds=k, shuffle=True)
        zero_kfold =  KFold(n=N_zeros, n_folds=k, shuffle=True)
        #this is how you go through all the training data
#         for train_index, test_index in kf:
# ...    print("TRAIN:", train_index, "TEST:", test_index)
# ...    X_train, X_test = X[train_index], X[test_index]
# ...    y_train, y_test = y[train_index], y[test_index]

    #fix define

    #def experiment(self, helper function):

    #def kfold_evaluate(self, )






    def sort_prediction_all(self, predictions):
        #predictions finds the ranking of the as well as their corresponding indexes
        #assume predictions comes in a one-dimensional array
        sorted_values = np.sort(predictions);
        index = np.argsort(predictions); #this sorts by increasing order
        return(sorted_values[::-1], index[::-1])
        #return ( sorted_values, index[:, ::-1]) #this sorts by decreasing order


    #This is average precision
    def ranked_precision(self, *kwargs):
        #kwargs = (predictions, truth, k)
        #does average precision for a person with 2 arguments
        #does precision for rank k with 3 arguments
        #finds the precision of the dataset at certain point k for a certain mall
        #note that the index for k starts at 1 rather the traditional 0. Because there is no 0th ranked item
        #obviously, truth an pre dictions must be the same
        predictions = kwargs[0]
        truth = kwargs[1]
        if(not (predictions.shape == truth.shape )):
            raise Exception("prediction and truth are not the same size")
        #sort predictions. This is necessary
        (sorted_pred, index) = self.sort_prediction_all(predictions)
        truth = truth[index];
        if(len(kwargs) ==2):
            total_precision = 0;
            each_precision = np.divide(np.cumsum(truth), np.arange(1, np.arange(1, len(truth)+1)))
            average_precision = np.dot(each_precision, truth)/np.sum(truth)
            #slower version
            # for i in  np.arange(0, (truth.shape)[0]):
            #     #easier to do cumsum and divide by a range and multiply by industries
            #     if( truth[i] == 1):
            #         precision_value = np.sum(truth[0:(i+1)])/(i+1)
            #         total_precision = total_precision + precision_value
            # n_truth_correct = np.sum(truth);
            # average_precision = total_precision/n_truth_correct;
            return(average_precision)
        if(len(kwargs) ==3):
            k = kwargs[2]
            return np.sum(truth[0:(k)])/k

    #assume that test_indices are tuples
    def map(self, X, X_predict, test_indices):
        #should be robust if it is for list and array
        #stable sorting on the indices
        index = np.argsort(test_indices[:, 0]);
        test_indices = test_indices[:, index] #probably wrong
        index = np.argsort(test_indices[:, 1]);
        test_indices = test_indices[:, index] #probably wrong
        #sorted_indices = sorted(test_indices,key=lambda x: x)
        n_users_in_test = 0
        val = 0
        for i in range(X.shape[1]):
            canidates = test_indices[test_indices[:, 1] == i, :]
            if canidates.shape[0] >0:
                val = val + self.ranked_precision(X_predict[canidates[0], canidates[1]], X[canidates[0], canidates[1]])
                n_users_in_test = n_users_in_test +1
        return val/n_users_in_test;

    def ranked_recall(self, predictions, truth, k):
        #finds the precision of the dataset at certain point k for a certain mall
        #note that the index for k starts at 1 rather the traditional 0. Because there is no 0th ranked item
        #obviously, truth an predictions must be the same
        if(not (predictions.shape == truth.shape )):
            raise Exception("prediction and truth are not the same size")
        #return error if predictions and truth are not the same
        #denominator is actually the amount of correct items. just sum
        n_truth_correct = np.sum(truth);
        (sorted_pred, index) = self.sort_prediction_all(predictions)
        truth = truth[index]; #reorganizing truth
        return np.sum(truth[0:(k)])/n_truth_correct
   # def sort_rec_testing_data(self, predictions):
    #sorts for the testing data

    def bipartite_projection(self, adj, item_allocation):
        #gives you
        #adj is assumed to be the binary item-user matrix where item are the rows and user are the column
        #see paper in bipartite network projection and personal recommendation for reference
        adj = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1]]);
        size = adj.shape;
        item_deg=np.dot(adj, np.ones((size[1], 1))) #this finds the degree of the nodes in the item set
        item_power = np.transpose(adj * (1/item_deg));
        user_deg = np.dot(np.ones((1, size[0])),adj);
        user_power = adj * (1/user_deg)
        W = np.dot(user_power, item_power) #This tells you how much weight that item i will give to j depending how similar it is to j
        recommend_items= np.dot(W, item_allocation); #This will give you the item distribution for a certain users
        return (recommend_items, W);

def bipartite_projection(adj, item_allocation):
    #gives you
    #adj is assumed to be the binary item-user matrix where item are the rows and user are the column
    #see paper in bipartite network projection and personal recommendation for reference
    adj = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1]]);
    size = adj.shape;
    item_deg=np.dot(adj, np.ones((size[1], 1))) #this finds the degree of the nodes in the item set
    item_power = np.transpose(adj * (1/item_deg));
    user_deg = np.dot(np.ones((1, size[0])),adj);
    user_power = adj * (1/user_deg)
    W = np.dot(user_power, item_power) #This tells you how much weight that item i will give to j depending how similar it is to j
    recommend_items= np.dot(W, item_allocation); #This will give you the item distribution for a certain users
    return (recommend_items, W);


cons =  dict([('iter_max', 50), ('lambda', 0), ('error', 1)])
# cons["lambda"] = .10
# cons["error"] = 1

test = one_class()
tuple_size = (300, 30)
R = np.random.choice([0, 1], size=tuple_size, p=[.95, .05])
d = 10
model = ProjectedGradientNMF(n_components=d, init='random', random_state=0)
model.fit(R)

V_init = (model.components_).T #the matrix comes out as a d*n matrix, so you have to do the transpose to fix it.
#V_init = .1 * np.random.randn(tuple_size[1], d) + 0
W = np.ones(tuple_size)
test.wlas(R, W, d, V_init, cons)
