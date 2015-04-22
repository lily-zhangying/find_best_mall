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
    def __init__(self,X, similarity_helper = None, feature_helper = None, score_helper = None, \
                 item_feat = None, user_feat = None, cluster=None):
        super(content, self).__init__(X)
        self.feature_helper = feature_helper
        self.score_helper = score_helper
        self.item_feat = item_feat
        self.user_feat = user_feat
        self.similarity_helper = similarity_helper

    def get_helper2(self, name, function):
        super(content, self).get_helper2(name, function)


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
            lol, user_transform = self.feature_helper(X=user_ratings, item_feat = self.item_feat[:, 1], user_feat = user_feat)

        #assume that the similarity matrix is
        S = pairwise_distances(item_transform, user_transform, self.similarity_helper)
        predicted_values = S
        predicted_values[np.asarray(user_ratings)] = 0
        result = np.argsort(predicted_values)
        return result[0:k]

    def fit(self, train_indices = None, test_indices = None):
        super(content, self).transform_training(train_indices, test_indices)#setting up training data
        # shape return the rows and colonms of the matrix
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
        return super(content,  self).score(truth_index)




def user_to_item(X_train, item_feat, user_feat, start, end):
    #creates a nice lambda function
    START = start
    END = end+1#remember to +1 as an offset
    #stores that mallls belong into
    #creating a new item_transform matrix
    # LONG_IND is the colomn index of the user feature matrix
    user_transform = user_feat[:, START:END]
    item_transform = np.zeros((X_train.shape[0], END - START))
    #possibly faster if you use a join and a group in pandas
    for i in np.arange(X_train.shape[0]): #go through all stores
        mall_indexes = (X_train[i, :] == 1) #finds all the malls that have store i
        store_features = user_feat[mall_indexes, : ][:, START:END] #get coordinates fast
        test = np.average(store_features, axis=0)
        item_transform[i, :]= test

    return (item_transform, user_transform)

#helper that extracts columns from a the mall matrix
def user_to_item_helper(start, end):
    return lambda X, item_feat, user_feat : user_to_item(X, item_feat, user_feat, start, end)

# X = np.array([[1, 1, 1, 1] , [1, 1, 0, 0], [1, 0, 1, 0]])
# user_feat = np.array([[1, 1, 1, 2, 3], [0, 0, 4, 5, 6], [1, 0, 7, 8, 9], [0,1 , 10, 11, 12]])
# item_feat = None
# fun = user_to_item_helper(2, 4)
# cosine = similarity.cosine()
# test = content(X, similarity_helper=cosine, user_feat=user_feat, item_feat=item_feat, feature_helper=fun)
# test.fit()