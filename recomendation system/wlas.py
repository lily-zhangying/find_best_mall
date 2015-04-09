__author__ = 'John'
import numpy as np
from sklearn.decomposition import ProjectedGradientNMF
import recsys
import evaluate
from sklearn import decomposition
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
        self.spareness = sparseness
        self.tol = tol

    def get_parameters(self, iter_max=100, sparseness=1, tol=1, n_topics =10):
        self.iter_max = iter_max
        self.spareness = sparseness
        self.tol = tol
        self.n_topics = n_topics

    def predict_for_user(self, user_ratings, user_feat, k, feature_transform_all =None):
        #feature_transform_all refers to items
        # shape return the rows and colonms of the matrix
        self.X = np.concatenate(self.X, user_ratings)
        Nrow, Ncol = self.X.shape
        if (feature_transform_all == None):
            if self.feature_helper == None:
                W = np.ones(Nrow, Ncol) #default
            else:
                W = self.feature_helper(self.X, np.concatenate(self.user_feat, user_feat), self.item_feat)
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
            W = self.feature_helper(self.X_train, self.user_feat, self.item_feat)

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

    def score(self, truth_index):
        super(wlas,  self).score(truth_index)

def uniform_weight(X, user_feat=None, item_feat=None, delta=.05):
    W=X
    W[W==0] = delta
    return W
#hi = np.zeros((5, 5))

#eggie = wlas(hi)
#eggie.