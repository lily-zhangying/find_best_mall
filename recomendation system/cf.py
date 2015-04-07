__author__ = 'John'
import numpy as np
from sklearn.decomposition import ProjectedGradientNMF
import recsys
import evaluate
import similarity
from sklearn import decomposition
from numpy.linalg import inv

#feature helper and user_feature are derived from lambda functions

class cf(recsys.recsys):
    def __init__(self,X, similarity_helper = None, feature_helper = None, score_helper = None, user_feat = None, cluster=None):
        super(cf, self).__init__(X)
        self.feature_helper = feature_helper
        self.score_helper = score_helper
        self.user_feat = user_feat
        self.similarity_helper = similarity_helper

    def get_parameters(self):
        pass

    def fit(self, train_indices):
        super(cf, self).transform_training(train_indices)
        Nitems, Nusers = self.X_train.shape
        self.X_predict = np.zeros((Nitems, Nusers))
        #unpack constants from dictionary here
        #setting constants

        #some how accomodate constants for two different constants
        #create the symmetric matrix

        #W represents a tranformed feature_helper function
        if self.feature_helper == None:
            W = self.user_feat
        else:
            W = self.feature_helper(X=self.X_train, feat = self.user_feat)

        #assume that the similarity matrix is
        S = self.similarity_helper(W)
        S = S-np.diag(S.diagonal())

        for i in range(Nitems):
            for j in range(Nusers):
                #do the average all of the users that are neighbors of j for item i
                #now, include clusters
                self.X_predict[i, j] = np.dot(S[j, :],self.X_train[i, :] )/np.sum(S[j, :])
        return self.X_predict

    def score(self, truth_index):
        super(cf,  self).score(truth_index)

