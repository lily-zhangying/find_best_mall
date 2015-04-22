__author__ = 'John'
import numpy as np
from sklearn.decomposition import ProjectedGradientNMF
import recsys
import cf
import evaluate
import similarity
from sklearn import decomposition
from numpy.linalg import inv
from nmf_analysis import mall_latent_helper as nmf_helper
from sklearn.metrics.pairwise import pairwise_distances

#feature helper and user_feature are derived from lambda functions

class cf_item(recsys.recsys):
    def __init__(self,X, similarity_helper = None, feature_helper = None, score_helper = None,\
                 item_feat = None, cluster=None, top_k = 30):
        super(cf_item, self).__init__(X)
        self.feature_helper = feature_helper
        self.score_helper = score_helper
        self.item_feat = item_feat
        self.similarity_helper = similarity_helper
        if(cluster == None):
            self.cluster = np.ones(X.shape[0])
        else:
            self.cluster = cluster
        self.feature_transform = None;
        self.top_k = top_k

    def get_parameters(self):
        pass

    def get_helper2(self, name, function):
        super(cf_item, self).get_helper2(name, function)


    def predict_for_user(self, user_ratings, k, feature_transform_all):
            #output: feature_transform_all is a similarity matrix in this case. remember that the ith column refers to one item and the rows represent how each item related to ith item

        Nitems, Nusers= self.X.shape
        S_norm = feature_transform_all


        predicted_values= np.dot(S_norm.T, user_ratings.reshape((Nitems, 1)))
        predicted_values[np.asarray(user_ratings)] = 0
        result = np.argsort(-1*predicted_values.T)
        return result[0:k]

    def fit(self, train_indices = None, test_indices = None):
        #using cf to do the calculations. basically, this is related to regular to regular cf just by doing the transpose of X and swapping some stuff
        model = cf.cf(self.X.T,user_feat=self.item_feat, feature_helper=self.feature_helper, similarity_helper=self.similarity_helper,\
              top_k=self.top_k, cluster=self.cluster\
            )
        if(not train_indices == None):
            train_indices_for_model = np.column_stack((train_indices[:, 1], train_indices[:, 0]))
        if(not test_indices == None):
            test_indices_for_model = np.column_stack((test_indices[:, 1], test_indices[:, 0]))
        self.X_predict = model.fit(train_indices, test_indices)
        self.X_predict = self.X_predict.T
        self.X_train = model.X_train.T


        return self.X_predict

    def score(self, truth_index):
        return super(cf_item,  self).score(truth_index)

# nmf = nmf_helper(2)
# X= np.array([[1, 0, 0, 0],[0, 1, 1, 1]]).T
# feat = np.array([[0, 0, 0, 1, 1, 1, 1], [7, 7, 7, 0, 0, 0,0], [7, 8, 7, 0, 0, 0, 0], [7, 7, 7, 0, 0, 0, 0]])
# cosine = similarity.cosine()
# lily = cf_item(X, similarity_helper = cosine, feature_helper=nmf, item_feat=feat)
# lily.predict_for_user(np.array([[0, 1, 1, 0]]), 1)



# first_5 = np.column_stack( (.1*np.random.randn(5, 4) + 20, np.zeros((5, 6))))
# last_5 = np.column_stack( (np.zeros((5, 6)), .1*np.random.randn(5, 4) + 100))
# other_last_3 = np.array([np.arange(10)]*3)
# feat = np.row_stack((first_5, last_5, np.array([np.arange(10)]), np.array([np.arange(10)]) +1, np.array([np.arange(10)]) +2))
# #, other_last_3
#
#
# cosine = similarity.cosine()
#
# X= np.array([[1, 1, 1, 0, 0, \
#               1, 1, 1, 1 ,0, \
#               1 ,1 ,1], \
#              [0, 1, 1, 1, 1, \
#                                                       0, 0, 0, 0, 0, 1, 1, 1]])
# X =X.T
#
# test = cf_item(X, item_feat=feat,  similarity_helper=cosine, top_k = 3)
# test.fit()