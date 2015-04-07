__author__ = 'lily'
import numpy as np
from sklearn.decomposition import ProjectedGradientNMF
import recsys
import evaluate
import similarity
from sklearn import decomposition
from numpy.linalg import inv

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

    def fit(self, train_indices):
        super(content, self).transform_training(train_indices)#setting up training data
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
        S = self.similarity_helper(item_transform, user_transform)

        S[self.X_train == 1] =0
        self.X_predict = S

    def score(self, truth_index):
        super(content,  self).score(truth_index)


