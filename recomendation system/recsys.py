import numpy as np
from sklearn.metrics.pairwise import pairwise_distances



class recsys(object):
    #X is the truth
    def __init__(self,X):
        self.X = X
        self.X_predict = None
        self.X_train = None
        pass
    #get the necessary helper functions to do an analysis. may require more parameters for derived classes
    def get_helpers(self, feature_func = None, similarity_func = None):
        if ( not(feature_func is None) or (self.feature_helper is None)):
            self.feature_helper = feature_func;
        if ( not(similarity_func is None) or (self.similarity_helper is None)):
            self.similarity_helper = similarity_func;

    def similarity(self, features, similarity_helper, cluster, k):
        #creates an N-by-N matrix where the i, j entry tell how the ith person is related to the jth person. the column is referring to one persn
        # this matrix is NOT SYMMETRIC
        # input
        # feature - matrix how you are going to compare the objects where you have N peop
        # similarity_helper -

        S=pairwise_distances(features, metric=similarity_helper)
        #S = self.similarity_helper(W)
        S = S-np.diag(S.diagonal())
        #modifies S for cluster information
        cluster_ind = np.array([cluster]*features.shape[0])
        S = np.multiply(S, 1*(cluster_ind == cluster_ind.T))
        #implement the neighborbased part. This is for better results. Get top K similar people for each user.        np.apply_along_axis(find_top_k, 0,S , k=k) #computations can be slow for this model
        print(np.nonzero(S[1, :]))
        S_norm =np.multiply(S, 1/np.sum(S, axis=0))  #fast multiplication

        S_norm[np.isnan(S_norm)]=0 #deals with nan problem. (Consider the instance that you are the only user and nobody is similar to you.)
        return S_norm


    def get_parameters(self, **kwargs):
        pass
        #this varies from learner to learner. Some learners do not have this because they do not have parameters to be tuned

    def get_parameters_2(self, **kwargs):
        pass


    def predict_for_user(self, user_ratings, user_feat, k, feature_transform_all =None):
        #output: predicted indices of the stores that are most liked by a user
        #f transform user into a more appropiate feature
        #makes a prediction for the user
        #for matrix factorization, preprocessing must be made. Specifically, user_feat and feat must be already defined
        Nitems, Nusers= self.X.shape
        if (feature_transform_all == None):
            #this deals with the mf_preprocessing issue
            self.X = np.concatenate((self.X, user_ratings));
            self.feature = np.concatenate((self.feature, user_feat))
            transformed_user = self.feature_helper(self.X, self.feature)[:, Nusers] #get the features of the user of interest
            self.X = self.X[:, 0:Nusers] #now reset X and feature back to normal
            self.feature = self.feature[:, 0:Nusers]
        else:
            self.feature_transform = feature_transform_all;
            transformed_user = self.feature_helper(user_ratings, user_feat)
        S=pairwise_distances(self.feature_transform, transformed_user, self.similarity_helper) #should be a 1-d array
        S = S.reshape((1, Nusers))
        #garuntees that the shape is row matrix
        #np.array([S,]*Nitems) #creates duplicates of S rowwise
        predicted_values = np.average(np.multiply(self.X, np.array([S,]*Nitems)))/np.sum(S)
        predicted_values[user_ratings == 1] = 0
        result = np.argsort(predicted_values)
        return result[0:k]





    def transform_training(self, train_indices,  test_indices):
        #train_incides must be a |Train_Data|-by-2 matrix.
        #train_indices come in tuples
        self.X_train = np.copy(self.X);
        if((test_indices is None) and (train_indices is None) ):
            return
        elif(not (test_indices is None)):
            self.X_train[test_indices[:, 0], test_indices[:, 1]]  = np.zeros((1, test_indices.shape[0]))
            return
        else:
            #create a binary matrix that
            Nitems, Nusers = self.X.shape
            test_indicator = np.ones((Nitems, Nusers))

            test_indicator[train_indices[:, 0], train_indices[:, 1]]  = np.zeros((1, train_indices.shape[0]))
            self.X_train[test_indicator == 1] = 0

    def fit(self, train_indices = "None", test_indices = "None"):
        pass
        #the code of the actual
        #i

    #in reality, this would not be used alot
    def predict(self, indices):
        if(not isinstance(indices, np.ndarray)):
            raise Exception("Dawg, your indices have to be an ndarray")
        return self.X_predict(indices[:, 0], indices[:, 1])

    def score(self, truth_index):
        if(not isinstance(truth_index, np.ndarray)):
            raise Exception("Dawg, your testing indices have to be an ndarray")
        return self.score_helper(self.X, self.X_predict, truth_index)
        #do ranked precision
        #first

    def get_helper2(self, name, function):
        if(name == 'feature_helper'):
            self.feature_helper = function
            return
        if(name == 'similarity_helper'):
            self.similarity_helper = function
            return
        if(name == 'score_helper'):
            self.score_helper = function
            return
        else:
            raise Exception("Cannot find feature function corresponding to the input name")

def find_top_k(x, k):
    #return an array where anything less than the top k values of an array is zero
    if( np.count_nonzero(x) <k):
        return x
    else:
        x[x < -1*np.partition(-1*x, k)[k]] = 0
        return x

#This will be used if the algorithm is too slow
def test_helper(x, k):
    fun = lambda i: find_top_k(x[:, i], k)
    return np.vectorize(fun)

#clean similarity matrix
# def cluster_neighbors(cluster, S):
#     fun = lambda i : (S[i, cluster[i]] = 0)
#     return np.vectorize(fun)

