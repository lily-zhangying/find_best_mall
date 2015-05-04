__author__ = 'John'
import numpy as np
from sklearn.decomposition import ProjectedGradientNMF
import recsys
import evaluate
import similarity
from nmf_analysis import mall_latent_helper as nmf_helper
from sklearn import decomposition
from numpy.linalg import inv
from sklearn.metrics.pairwise import pairwise_distances

#feature helper and user_feature are derived from lambda functions

class cf(recsys.recsys):
    def __init__(self,X, similarity_helper = None, feature_helper = None, score_helper = None, \
                 user_feat = None, cluster=None, top_k = 30):
        super(cf, self).__init__(X)
        self.feature_helper = feature_helper
        self.score_helper = score_helper
        self.user_feat = user_feat
        self.similarity_helper = similarity_helper
        self.top_k = top_k
        if(cluster == None):
            self.cluster = np.ones(X.shape[1])
        else:
            self.cluster = cluster
        self.feature_transform = None;

    def get_helpers(self, feature = None, similar = None):
        if ( not(feature == None) or (self.feature_helper == None)):
            self.feature_helper = feature;
        if ( not(similar == None) or (self.similarity_helper== None)):
            self.similarity_helper = similar;

    #a way to get functions easily, one by one
    def get_helper2(self, name, function):
        super(cf, self).get_helper2(name, function)

    def remove_helpers(self, list):
        if "similar" in list:
            self.similarity_helper = None
        if "feature" in list:
            self.feature_helper = None

    def predict_for_user(self, user_ratings, user_feat, k, feature_transform_all):
        #output: predicted indices of the stores that are most liked by a user
        #feature_transform_all is the final transformation of some process that transforms user_ratings
        #feature transform user_rating should be the final transform rating
        Nitems, Nusers= self.X.shape

        user_averages = np.average(self.X, axis=0)
        average_matrix = np.array([user_averages]*Nitems) #create a average matrix for computation

        if (feature_transform_all is None):
            # #this deals with the matrix factorization prepossessing issue. deals with -b + 2a
            if self.feature_helper is None:
                self.feature_transform = self.user_feat #readily use the features
            else:
                self.feature_transform = self.feature_helper(np.concatenate((self.X, user_ratings), axis=1), np.concatenate((self.user_feat, user_feat)))
                transformed_user = self.feature_transform[Nusers, :] #get the features of the user of interest
                self.feature_transform = self.feature_transform[0:Nusers, :]
        else:
            self.feature_transform = feature_transform_all;
            if self.feature_helper is None:
                transformed_user = user_feat
            else:
                transformed_user = self.feature_helper(user_ratings, user_feat) #transform the users feature

        S=pairwise_distances(self.feature_transform, transformed_user, self.similarity_helper) #should be a 1-d array
        #print(S)
        S=S.reshape((Nusers, 1))/np.sum(S) #create a column vector
        #recsys.find_top_k(S, self.top_k)
        np.apply_along_axis(recsys.find_top_k, 0,S , k=k)


        self.predicted_values = np.dot(self.X-average_matrix, S) + np.average(user_ratings)#simple matrix-vector multiplication
        #self.predicted_values[np.asarray(user_ratings)] = 0
        self.predicted_values[np.nonzero(user_ratings)] = 0
        result = np.argsort(-1*self.predicted_values.T) #predict top results
        result = result.reshape(-1)
        return result[0:k]



    def get_parameters(self):
        pass

    def fit(self, train_indices = None, test_indices = None):
        #initializing
        super(cf, self).transform_training(train_indices, test_indices)
        Nitems, Nusers = self.X_train.shape

        #feature transformation
        if self.feature_helper == None:
            self.feature_transform = self.user_feat
        else:
            self.feature_transform = self.feature_helper(X=self.X_train, feat = self.user_feat)
        #print(self.feature_transform)

        #compute similarity between items
        S_norm = super(cf, self).similarity(self.feature_transform,self.similarity_helper, self.cluster, self.top_k)
        #print(np.sum(S_norm))

        user_averages = np.average(self.X_train, axis=0)
        average_matrix = np.array([user_averages]*Nitems) #create a average matrix for computation

        self.X_predict = average_matrix+ np.dot(self.X_train-average_matrix, S_norm ) #compute X_predict fast
        self.X_predict[self.X_train == 1] =1

        return self.X_predict

    def score(self, truth_index):
        return super(cf,  self).score(truth_index)

#feature transformation is not needed for test
#test for using predict_for_user
# cosine = similarity.cosine()
# nmf = nmf_helper(2)
# X= np.array([[0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 0]]).T
# feat = np.array([[7, 7, 7, 0, 0, 0,0], [7, 8, 7, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1]])
# test = cf(X, feature_helper=nmf, user_feat=feat, similarity_helper=cosine)
# test.predict_for_user(np.array([[0, 1, 0, 1]]).T, np.array([[7, 7, 7, 0, 0, 0, 0]]), 1)

#
#
# first_5 = np.column_stack( (.1*np.random.randn(5, 4) + 20, np.zeros((5, 6))))
# last_5 = np.column_stack( (np.zeros((5, 6)), .1*np.random.randn(5, 4) + 100))
# other_last_3 = np.array([np.arange(10)]*3)
# feat = np.row_stack((first_5, last_5, np.array([np.arange(10)]), np.array([np.arange(10)]) +1, np.array([np.arange(10)]) +2))
# #, other_last_3
#
#
# cosine = similarity.cosine()
# nmf = nmf_helper(2)
# X= np.array([[1, 1, 1, 0, 0, \
#               1, 1, 1, 1 ,0, \
#               1 ,1 ,1], \
#              [0, 1, 1, 1, 1, \
#                                                       0, 0, 0, 0, 0, 1, 1, 1]])
#
# test = cf(X, user_feat=feat,  similarity_helper=cosine, top_k = 3)
# test.fit()
# user_ratings = np.array([0, 0])
# user_feat = np.array([20, 20, 20, 20, 0, 0, 0, 0,0, 0, 0, 0, 0])
# k = 2
# #feature_transform_all = feat
# #test.predict_for_user(user_ratings, user_feat, k, feature_transform_all)