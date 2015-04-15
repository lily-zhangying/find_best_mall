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

class pop_rec(recsys.recsys):
    def __init__(self,X, similarity_helper = None, feature_helper = None, score_helper = None, user_feat = None, cluster=None):
        super(pop_rec, self).__init__(X)
        self.feature_helper = feature_helper
        self.score_helper = score_helper
        self.user_feat = user_feat
        self.similarity_helper = similarity_helper
        if(cluster == None):
            self.cluster = np.ones((X.shape[1],1))
        else:
            self.cluster = cluster
        self.feature_transform = None;

    def get_helpers(self, feature = None, similar = None):
        if ( not(feature == None) or (self.feature_helper == None)):
            self.feature_helper = feature;
        if ( not(similar == None) or (self.similarity_helper== None)):
            self.similarity_helper = similar;
    def remove_helpers(self, list):
        if "similar" in list:
            self.similarity_helper = None
        if "feature" in list:
            self.feature_helper = None

    def predict_for_user(self, user_ratings, user_feat, k, feature_transform_all =None):
        #output: predicted indices of the stores that are most liked by a user
        #feature_transform_all is the final transformation of some processee.
        Nitems, Nusers= self.X.shape
        # if (feature_transform_all == None):
        #     # #this deals with the matrix factorization prepossessing issue. deals with -b + 2a
        #     if self.feature_helper == None:
        #         self.feature_transform = self.user_feat #readily use the features
        #     else:
        #         self.feature_transform = self.feature_helper(np.concatenate((self.X, user_ratings), axis=1), np.concatenate((self.user_feat, user_feat)))
        #         transformed_user = self.feature_transform[Nusers, :] #get the features of the user of interest
        #         self.feature_transform = self.feature_transform[0:Nusers, :]
        # else:
        #     self.feature_transform = feature_transform_all;
        #     transformed_user = self.feature_helper(user_ratings, user_feat) #transform the users feature


        predicted_values = np.sum(self.X, axis=1)/Nusers

        predicted_values[np.asarray(user_ratings)] = 0
        result = np.argsort(-1*predicted_values.T) #predict top results
        return result[0:k]

    def get_parameters(self):
        pass

    def fit(self, train_indices = None, test_indices = None):
        #initializing
        super(pop_rec, self).transform_training(train_indices, test_indices)
        Nitems, Nusers = self.X_train.shape
        self.X_predict = np.zeros((Nitems, Nusers))

        popularity = np.sum(self.X_train, axis=1)/Nusers
        self.X_predict = np.array([popularity]*Nusers).transpose()
        self.X_predict[self.X_train == 1] = 1



        return self.X_predict

    def score(self, truth_index):
        super(pop_rec,  self).score(truth_index)



cosine = similarity.cosine()
nmf = nmf_helper(2)
X= np.array([[1, 0, 0, 0],[0, 1, 1, 1], [0, 1, 1, 0]])
feat = np.array([[0, 0, 0, 1, 1, 1, 1], [7, 7, 7, 0, 0, 0,0], [7, 8, 7, 0, 0, 0, 0], [7, 7, 7, 0, 0, 0, 0]])

test = cf(X, feature_helper=nmf, user_feat=feat, similarity_helper=cosine)
test.fit(test_indices =np.array([[2, 3]]))