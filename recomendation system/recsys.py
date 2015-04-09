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
        if ( not(feature_func == None) or (self.feature_helper == None)):
            self.feature_helper = feature_func;
        if ( not(similarity_func == None) or (self.similarity_helper== None)):
            self.similarity_helper = similarity_func;

    def get_parameters(self, **kwargs):
        pass
        #this varies from learner to learner. Some learners do not have this because they do not need to get learned

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
        self.X_train = self.X;
        if((test_indices== None) and (train_indices == None) ):
            return
        if(not (test_indices== None)):
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



