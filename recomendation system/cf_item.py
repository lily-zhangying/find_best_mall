__author__ = 'John'
import numpy as np
from sklearn.decomposition import ProjectedGradientNMF
import recsys
import evaluate
import similarity
from sklearn import decomposition
from numpy.linalg import inv
from nmf_analysis import mall_latent_helper as nmf_helper
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
            self.cluster = np.ones((X.shape[0],1))
        else:
            self.cluster = cluster
        self.feature_transform = None;

    def get_parameters(self):
        pass

    def predict_for_user(self, user_ratings, k, feature_transform_all =None):
            #output: feature_transform_all is a similarity matrix in this case
        Nitems, Nusers= self.X.shape
        if (feature_transform_all == None):

            if self.feature_helper == None:
                self.feature_transform = self.item_feat #readily use the features
            else:
                self.feature_transform = self.feature_helper(X=self.X_train, feat = self.item_feat)

            #assume that the similarity matrix is
            S=pairwise_distances(self.feature_transform, metric=self.similarity_helper)
        else:
            S = feature_transform_all
        S = S-np.diag(S.diagonal())
        S_norm = np.multiply(S.T, 1/np.sum(S, axis=1).T).T
        S_norm[np.isnan(S_norm)]=0 #deals with nan problem
        #         #modifies S for cluster information
        for i in range(Nitems):
            cluster_ind = 1*(self.cluster == self.cluster[i])#binary vector indicating all the neighbors of i
            S[i, :]= np.multiply(cluster_ind.T, S[i, :])
        predicted_values= np.dot(S_norm, user_ratings.reshape((Nitems, 1)))
        predicted_values[np.asarray(user_ratings)] = 0
        result = np.argsort(-1*predicted_values.T)
        return result[0:k]

    def fit(self, train_indices = None, test_indices = None):
        #initializing
        super(cf_item, self).transform_training(train_indices, test_indices)#setting up training data
        Nitems, Nusers = self.X_train.shape
        self.X_predict = np.zeros((Nitems, Nusers))


        #W represents a tranformed feature_helper function
        if self.feature_helper == None:
            self.feature_transform = self.item_feat
        else:
            self.feature_transform = self.feature_helper(X=self.X_train, feat = self.item_feat)

        #assume that the similarity matrix is
        S=pairwise_distances(self.feature_transform, metric=self.similarity_helper)
        S = S-np.diag(S.diagonal())
        #modifies S for cluster information for items
        for i in range(Nitems):
            cluster_ind = 1*(self.cluster == self.cluster[i])#binary vector indicating all the neighbors of i
            S[i, :]= np.multiply(cluster_ind.T, S[i, :])
        S_norm = np.multiply(S.T, 1/np.sum(S, axis=1).T).T
        S_norm[np.isnan(S_norm)]=0 #deals with nan problem
        self.X_predict = np.dot(S_norm, self.X_train )
        self.X_predict[self.X_train == 1] =1
        return self.X_predict

    def score(self, truth_index):
        super(cf_item,  self).score(truth_index)

nmf = nmf_helper(2)
X= np.array([[1, 0, 0, 0],[0, 1, 1, 1]]).T
feat = np.array([[0, 0, 0, 1, 1, 1, 1], [7, 7, 7, 0, 0, 0,0], [7, 8, 7, 0, 0, 0, 0], [7, 7, 7, 0, 0, 0, 0]])
cosine = similarity.cosine()
lily = cf_item(X, similarity_helper = cosine, feature_helper=nmf, item_feat=feat)
lily.predict_for_user(np.array([[0, 1, 1, 0]]), 1)
