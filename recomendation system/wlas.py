__author__ = 'John'
import numpy as np
from sklearn.decomposition import ProjectedGradientNMF
import recsys
import evaluate
import content
import similarity
from sklearn import decomposition
from sklearn.metrics.pairwise import pairwise_distances
from numpy.linalg import inv


class wlas(recsys.recsys):
    def __init__(self,X, feature_helper = None, score_helper = None, user_feat = None, item_feat = None,
    iter_max = 100, sparseness =1, tol =1, n_topics = 10):
        super(wlas, self).__init__(X)
        self.feature_helper = feature_helper
        self.score_helper = score_helper
        self.user_feat = user_feat
        self.item_feat = item_feat
        self.iter_max = iter_max
        self.sparseness = sparseness
        self.tol = tol
        self.n_topics = n_topics

    def get_parameters(self, iter_max=100, sparseness=1, tol=1, n_topics =10):
        self.iter_max = iter_max
        self.sparseness = sparseness
        self.tol = tol
        self.n_topics = n_topics

    def get_parameters_2(self, kwargs):
        for key, value in kwargs.items():
            if(key == 'iter_max'):
                self.iter_max = value
            elif(key == 'sparseness'):
                self.sparseness = value
            elif(key == 'tol'):
                self.tol = value
            elif(key == "n_topics"):
                self.n_topics = value
            else:
                raise Exception("Not a valid parameter for the model")

    def get_helper2(self, name, function):
        super(wlas, self).get_helper2(name, function)


    def predict_for_user(self, user_ratings, user_feat, k, feature_transform_all =None):
        #feature_transform_all refers to items
        # shape return the rows and colonms of the matrix
        self.X = np.concatenate(self.X, user_ratings)
        Nrow, Ncol = self.X.shape
        if (feature_transform_all == None):
            if self.feature_helper == None:
                W = np.ones(Nrow, Ncol) #default
            else:
                W = self.feature_helper(self.X, self.item_feat, np.concatenate(self.user_feat, user_feat))
        else:
            W = feature_transform_all
        n_topics = self.n_topics
        nmf_data= decomposition.NMF(n_components=n_topics, sparseness='components', beta=1).fit(self.X)
        V_init = nmf_data.components_.T
        U = np.zeros((Nrow, n_topics))
        I = np.eye(n_topics)
        V = V_init



        for k in range(self.iter_max):
            for i in range(Nrow):
                W_hat_i = np.diag(W[i, :])
                (V.T).dot(W_hat_i)
                psm = (V.T).dot(W_hat_i).dot(V) +self.sparseness*sum(W[i, :])*I #the long matrix in the paper that is claimed to be semi positive definite
                U[i, :] = self.X[i, :].dot(W_hat_i).dot(V).dot(inv(psm))
            for j in range(Ncol):
                W_hat_j = np.diag(W[:, j])
                psm = U.T.dot(W_hat_j).dot( U) + self.sparseness*sum(W[:, j])*I
                V[j, :] = (self.X[:, j].T).dot(W_hat_j).dot(U).dot(inv(psm))
            #computing error
            diff = self.X - np.dot(U, V.T)
            diff_square = np.multiply(diff, diff)
            error = sum(sum(np.multiply(W, diff_square)))
            if(error < self.tol):
                break
        predicted_values = np.dot(U, V.T)[Nrow]
        predicted_values[np.asarray(user_ratings)] = 0
        result = np.argsort(predicted_values)
        return result[0:k]


    def fit(self, train_indices=None, test_indices = None):
        super(wlas, self).transform_training(train_indices, test_indices)
        Nrow, Ncol = self.X_train.shape
        #unpack constants from dictionary here
        #setting constants
        if self.feature_helper == None:
            W = np.ones(Nrow, Ncol) #default
        else:
            W = self.feature_helper(self.X_train, self.item_feat, self.user_feat)

        n_topics = self.n_topics
        nmf_data= decomposition.NMF(n_components=n_topics, sparseness='components', beta=1).fit(self.X)
        V_init = nmf_data.components_.T
        U = np.zeros((Nrow, n_topics))
        I = np.eye(n_topics)
        V = V_init



        for k in range(self.iter_max):
            for i in range(Nrow):
                W_hat_i = np.diag(W[i, :])
                psm = (V.T).dot(W_hat_i).dot(V) +self.sparseness*sum(W[i, :])*I #the long matrix in the paper that is claimed to be semi positive definite
                U[i, :] = self.X_train[i, :].dot(W_hat_i).dot(V).dot(inv(psm))
            for j in range(Ncol):
                W_hat_j = np.diag(W[:, j])
                psm = U.T.dot(W_hat_j).dot( U) + self.sparseness*sum(W[:, j])*I
                V[j, :] = (self.X_train[:, j].T).dot(W_hat_j).dot(U).dot(inv(psm))
            #computing error
            diff = self.X_train - np.dot(U, V.T)
            diff_square = np.multiply(diff, diff)
            error = sum(sum(np.multiply(W, diff_square)))
            if(error < self.tol):
                break
        self.X_predict = np.dot(U, V.T)
        self.X_predict[self.X_train == 1] =1
        return self.X_predict


    def score(self, truth_index):
        return super(wlas,  self).score(truth_index)

def content_based_weight_auxillary(X, item_feat, user_feat, feature_helper, similarity_helper):
    #Similarity and feature helper
    #pairwise_distances(item_transform, user_transform, similarity_helper)
    item_transform, user_transform = feature_helper(X=X, item_feat = item_feat, user_feat = user_feat)
    S =  pairwise_distances(item_transform, user_transform, similarity_helper)
    return 1- S

def content_based_weight(feature_helper,similarity_helper):
    return lambda X, item_feat, user_feat: content_based_weight_auxillary(X, item_feat,  user_feat,feature_helper, similarity_helper)


def user_weight(X, item_feat=None, user_feat=None ):
    #find the column sum
    sum_weight = np.sum(X,axis=0)
    sum_weight = sum_weight/X.shape[0] #normalize

    #values within a column are the same
    #generate the W matrix for each column. So duplicate rowwise by the number of items X.shape[0]
    W = np.array([sum_weight]*X.shape[0])
    W[X==1] = 1
    return W+.0000000001

def item_weight(X, item_feat=None, user_feat=None ):
    #find the column sum
    sum_weight = np.sum(X,axis=1)
    sum_weight = X.shape[1] - sum_weight
    sum_weight = sum_weight/X.shape[1] #normalize
    #values within a row are the same
    #generate the W matrix for each column. So duplicate rowwise by the number of items X.shape[0]
    W = np.array([sum_weight]*X.shape[1]).T
    W[X ==1] = 1
    return W

def uniform_weight(X,item_feat=None,  user_feat=None,  delta=.2):
    W=X
    W[W==0] = delta
    return W






# X = np.array([[1, 1, 1, 1] , [1, 1, 0, 0], [1, 0, 1, 0]])
# user_feat = np.array([[1, 1, 1, 2, 3], [0, 0, 4, 5, 6], [1, 0, 7, 8, 9], [0,1 , 10, 11, 12]])
# item_feat = None
# fun = content.user_to_item_helper(2, 4)
#
# cosine = similarity.cosine()
# content_helper = content_based_weight(fun,cosine)
#
#
#
#
# test = wlas(X, feature_helper = content_helper, user_feat=user_feat, item_feat=item_feat, n_topics=2)
# test.fit()