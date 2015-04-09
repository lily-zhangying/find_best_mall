__author__ = 'John'
import numpy as np
from sklearn.decomposition import ProjectedGradientNMF
import recsys
import evaluate
import similarity
from sklearn import decomposition
from numpy.linalg import inv
from sklearn.metrics.pairwise import pairwise_distances

#feature helper and user_feature are derived from lambda functions

class cf_item(recsys.recsys):
    def __init__(self,X, similarity_helper = None, feature_helper = None, score_helper = None, item_feat = None, cluster=None):
        super(cf_item, self).__init__(X)
        self.feature_helper = feature_helper
        self.score_helper = score_helper
        self.item_feat = item_feat
        self.similarity_helper = similarity_helper
        if(cluster == None):
            self.cluster = np.ones((X.shape[1],1))
            #all things belong into the same cluster

    def get_parameters(self):
        pass

    def predict_for_user(self, user_ratings, k, feature_transform_all =None):
            #output: feature_transform_all is a similarity matrix in this case
        Nitems, Nusers= self.X.shape
        if (feature_transform_all == None):

            if self.feature_helper == None:
                self.feature_transform = self.item_feat
            else:
                self.feature_transform = self.feature_helper(X=self.X_train, feat = self.item_feat)

            #assume that the similarity matrix is
            S=pairwise_distances(self.feature_transform, self.similarity_helper)
        else:
            S = feature_transform_all
        S = S-np.diag(S.diagonal())
        predicted_values= np.dot(S, user_ratings.reshape((Nitems, 1)))
        #modifies S for cluster information
        # for i in range(Nusers):
        #     cluster_ind = 1*(self.cluster == self.cluster[i])#binary vector indicating all the neighbors of i
        #     S[i, :]= np.multiply(cluster_ind, S[i, :])
        # predicted_values = np.asarray(np.zeros((1, Nitems)))
        # for i in range(Nitems):
        #
        #     if(not (self.user_ratings[i] ==0)):
        #         self.user_ratings[i] = np.dot(S[i, :], self.user_ratings)/np.sum(S[i, :])

        predicted_values[np.asarray(user_ratings)] = 0
        result = np.argsort(predicted_values)
        return result[0:k]

    def fit(self, train_indices = None, test_indices = None):
        super(cf_item, self).transform_training(train_indices, test_indices)#setting up training data
        # shape return the rows and colonms of the matrix
        Nitems, Nusers = self.X_train.shape
        self.X_predict = np.zeros((Nitems, Nusers))
        #unpack constants from dictionary here
        #setting constants

        #some how accomodate constants for two different constants
        #create the symmetric matrix

        #W represents a tranformed feature_helper function
        if self.feature_helper == None:
            self.feature_transform = self.item_feat
        else:
            self.feature_transform = self.feature_helper(X=self.X_train, feat = self.item_feat)

        #assume that the similarity matrix is
        S=pairwise_distances(self.feature_transform, self.similarity_helper)
        S = S-np.diag(S.diagonal())
        #modifies S for cluster information

        for i in range(Nusers):
            cluster_ind = 1*(self.cluster == self.cluster[i])#binary vector indicating all the neighbors of i
            S[i, :]= np.multiply(cluster_ind, S[i, :])
        S_norm = np.multiply(S.T, 1/np.sum(S, axis=1).T).T #check correctness
        self.X_predict = np.dot(S_norm, self.X_train )
        self.X_predict[self.X_train == 1] =0
        # for i in range(Nitems):
        #     for j in range(Nusers):
        #         #do the average all of the users that are neighbors of j for item i
        #         #now, include clusters
        #         # self.X_predict[i, j] = np.dot(S[j, :],self.X_train[i, :] )/np.sum(S[j, :])
        #         if(self.X_train[i, j] ==1):
        #             self.X_predict[i, j] = 1
        #         else:
        #             self.X_predict[i, j] = np.dot(S[i, :], self.X_train[:, j])/np.sum(S[i, :])
        return self.X_predict

    def score(self, truth_index):
        super(cf_item,  self).score(truth_index)

X = np.array([[1, 1, 0], [1, 1, 0]])
lily = cf_item(X, similarity_helper = similarity.test)
test_indices = np.array([[0, 0], [1, 1]])
lily.fit(test_indices = test_indices)
