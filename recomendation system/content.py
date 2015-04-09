__author__ = 'lily'
import numpy as np
from sklearn.decomposition import ProjectedGradientNMF
import recsys
import evaluate
import similarity
from sklearn import decomposition
from numpy.linalg import inv
from sklearn.metrics.pairwise import pairwise_distances

#feature helper and user_feature are derived from lambda functions

class content(recsys.recsys):
    def __init__(self,X, similarity_helper = None, feature_helper = None, score_helper = None, item_feat = None, user_feat = None, cluster=None):
        super(content, self).__init__(X)
        self.feature_helper = feature_helper
        self.score_helper = score_helper
        self.item_feat = item_feat
        self.user_feat = user_feat
        self.similarity_helper = similarity_helper

    def get_parameters(self):
        pass


    def predict_for_user(self, user_ratings, user_feat, k, feature_transform_all =None):
        #feature_transform_all refers to items
        # shape return the rows and colonms of the matrix
        Nitems, Nusers = self.X.shape
        #W represents a tranformed feature_helper function
        if (feature_transform_all == None):
            if self.feature_helper == None:
                item_transform = self.item_feat
                user_transform = user_feat
            else:
                item_transform, user_transform = self.feature_helper(X=user_ratings, item_feat = self.item_feat, user_feat = user_feat)
        else:
            item_transform= feature_transform_all
            lol, user_transform = self.feature_helper(X=user_ratings, item_feat = self.item_feat[:, 0], user_feat = user_feat)

        #assume that the similarity matrix is
        S = pairwise_distances(item_transform, user_transform, self.similarity_helper)
        predicted_values = S
        predicted_values[np.asarray(user_ratings)] = 0
        result = np.argsort(predicted_values)
        return result[0:k]

    def fit(self, train_indices = None, test_indices = None):
        super(content, self).transform_training(train_indices, test_indices)#setting up training data
        # shape return the rows and colonms of the matrix
        Nitems, Nusers = self.X_train.shape
        #unpack constants from dictionary here
        #setting constants

        #some how accomodate constants for two different constants
        #create the symmetric matrix

        #W represents a tranformed feature_helper function
        if self.feature_helper == None:
            item_transform = self.item_feat
            user_transform = self.user_feat
        else:
            item_transform, user_transform = self.feature_helper(X=self.X_train, item_feat = self.item_feat, user_feat = self.user_feat)

        #assume that the similarity matrix is
        S = pairwise_distances(item_transform, user_transform, self.similarity_helper)
        S[self.X_train == 1] =1
        self.X_predict = S

    def score(self, truth_index):
        super(content,  self).score(truth_index)


# def distance(X_train, item_feat, user_feat):
#     LONG_IND =
#     LAD_IND =
#     #stores that mallls belong into
#     #creating a new item_transform matrix
#     # LONG_IND is the colomn index of the user feature matrix
#     user_transform = user_feat[:, (LONG_IND, LAD_IND)]
#     item_transform = np.zeros((X_train.shape[0], 2))
#     for i in np.arange(X_train.shape[0]):
#         mall_indexes = (X_train[i, :] == 1)
#         stores_coordinates = user_feat[mall_indexes, : ][:, (LONG_IND, LAD_IND)] #get coordinates fast
#         item_transform[i, :]= np.mean(stores_coordinates, axis=0)
#
#     return (item_transform, user_transform)