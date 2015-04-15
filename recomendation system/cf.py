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
    def __init__(self,X, similarity_helper = None, feature_helper = None, score_helper = None, user_feat = None, cluster=None):
        super(cf, self).__init__(X)
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
        if (feature_transform_all == None):
            # #this deals with the matrix factorization prepossessing issue. deals with -b + 2a
            if self.feature_helper == None:
                self.feature_transform = self.user_feat #readily use the features
            else:
                self.feature_transform = self.feature_helper(np.concatenate((self.X, user_ratings), axis=1), np.concatenate((self.user_feat, user_feat)))
                transformed_user = self.feature_transform[Nusers, :] #get the features of the user of interest
                self.feature_transform = self.feature_transform[0:Nusers, :]
        else:
            self.feature_transform = feature_transform_all;
            transformed_user = self.feature_helper(user_ratings, user_feat) #transform the users feature
        S=pairwise_distances(self.feature_transform, transformed_user, self.similarity_helper) #should be a 1-d array
        S=S.reshape((Nusers, 1))/np.sum(S) #create a column vector
        predicted_values = np.dot(X, S)#simple matrix-vector multiplication
        S = np.squeeze(np.asarray(S)) #makes S into an aray
        #predicted_values = np.asarray(np.sum(np.multiply(self.X, np.array([S,]*Nitems)), axis=1)/np.sum(S)) old version which is complicated
        predicted_values[np.asarray(user_ratings)] = 0
        result = np.argsort(-1*predicted_values.T) #predict top results
        return result[0:k]

    def get_parameters(self):
        pass

    def fit(self, train_indices = None, test_indices = None):
        #initializing
        super(cf, self).transform_training(train_indices, test_indices)
        Nitems, Nusers = self.X_train.shape
        self.X_predict = np.zeros((Nitems, Nusers))

        #feature transformation
        if self.feature_helper == None:
            self.feature_transform = self.user_feat
        else:
            self.feature_transform = self.feature_helper(X=self.X_train, feat = self.user_feat)

        #assume that the similarity matrix is
        S=pairwise_distances(self.feature_transform, metric=self.similarity_helper)
        #S = self.similarity_helper(W)
        S = S-np.diag(S.diagonal())
        #modifies S for cluster information

        for i in range(Nusers):
            cluster_ind = 1*(self.cluster == self.cluster[i])#binary vector indicating all the neighbors of i
            S[i, :]= np.multiply(cluster_ind.T, S[i, :])
        S_norm =np.multiply(S, 1/np.sum(S, axis=0))  #fast multiplication
        S_norm[np.isnan(S_norm)]=0 #deals with nan problem
        self.X_predict = np.dot(self.X_train, S_norm )
        self.X_predict[self.X_train == 1] =1 #compute X_predict fast, slow version is below. It's left for understanding things
        #old version, which is slow and not concise
        # for i in range(Nitems):
        #     for j in range(Nusers):
        #         #do the average all of the users that are neighbors of j for item i
        #         #now, include clusters
        #         #may want to consider the case where the matrix is already 1. Then set to 0
        #         if(self.X_train[i, j] ==1):
        #             self.X_predict[i, j] = 1
        #         else:
        #
        #             self.X_predict[i, j] = np.dot(S[j, :],self.X_train[i, :] )/np.sum(S[j, :])
        return self.X_predict

    def score(self, truth_index):
        super(cf,  self).score(truth_index)

#feature transformation is not needed for test
#test for using predict_for_user
# cosine = similarity.cosine()
# nmf = nmf_helper(2)
# X= np.array([[0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 0]]).T
# feat = np.array([[7, 7, 7, 0, 0, 0,0], [7, 8, 7, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1]])
# test = cf(X, feature_helper=nmf, user_feat=feat, similarity_helper=cosine)
# test.predict_for_user(np.array([[0, 1, 0, 1]]).T, np.array([[7, 7, 7, 0, 0, 0, 0]]), 1)

cosine = similarity.cosine()
nmf = nmf_helper(2)
X= np.array([[1, 0, 0, 0],[0, 1, 1, 1], [0, 1, 1, 0]])
feat = np.array([[0, 0, 0, 1, 1, 1, 1], [7, 7, 7, 0, 0, 0,0], [7, 8, 7, 0, 0, 0, 0], [7, 7, 7, 0, 0, 0, 0]])

test = cf(X, feature_helper=nmf, user_feat=feat, similarity_helper=cosine)
test.fit(test_indices =np.array([[2, 3]]))