__author__ = 'John'
import numpy as np
from sklearn.decomposition import ProjectedGradientNMF
import recsys
import evaluate
import similarity
from nmf_analysis import mall_latent_helper as nmf_helper
from sklearn import decomposition
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv
from sklearn.metrics.pairwise import pairwise_distances
import random
import pop_rec

#feature helper and user_feature are derived from lambda functions

class ensemble(recsys.recsys):
    def __init__(self,models, fun, score_helper = None, train_index = None, test_index= None):
        #models is a list. assumed that they have helper functions attached to them.
        self.X = models[0].X
        self.score_helper = score_helper
        self.models = models
        self.fun = fun
        self.train_index = train_index
        self.test_index = test_index
        #insert actual helper here
    #
    # def train_validation_test_split(self, train_index, test_index):
    #     # using the
    #     pass

    #models, validation_indices, score_helper = None
    def train_models(self, train_index, validation_index):
        #combine validation_index and train_index together.
        combined_indices = np.vstack(validation_index, train_index)
        for i in len(self.models):
            self.models[i].fit(combined_indices)

    def cross_validation(self, k):
        #fun is the function that returns the optimal B for any creating the ensemble.
        #retuns the average B for an ensemble algorithm
        kf = cross_validation.KFold(n = len(self.train), n_folds=k)
        B = np.zeros(len(self.train), k)
        i= 0
        for local_train_index, local_validation_index  in kf:
            #train the models
            print("Computing for fold %s/%s" %(i, k))
            self.local_train = local_train_index
            self.local_validation = local_validation_index
            self.train_models(local_train_index, local_validation_index)
            ensembler = self.fun(self.models,local_validation_index, self.score_helper ) #or use the ensemble method
            B[:, i] =  ensembler.fit()
            i=i+1
            #run the algorithms.
            #Find the average
        B_average = np.average(B, axis=1)
        return B_average


    def compute_model(self, B):
        #obtains a linear combination of the X predict matrix
        self.X_predict = np.zeros(self.X.shape)
        for i in range(len(self.models)): #maybe could be faster
            self.X_predict = self.X_predict + B[i] *(self.models[i].X_predict - self.mu[i])/self.sigma[i]

    def score(self, truth_index):
        return super(ensemble,  self).score(truth_index)




class SA(recsys.recsys):

    def __init__(self,models, validation_indices, score_helper = None, eps = .001):
    #def __init__(self,models, train_indices, validation_indices, test_indices, score_helper = None, eps = .01):

        #indices are the actual indices of the matrix X
    #simulated annealing returns B
    #return B
        self.X = models[0].X
        self.score_helper = score_helper
        self.models = models
        #self.train_indices = train_indices
        self.validation_indices = validation_indices
        self.eps = eps

    def normalize_values(self):
        #obtain the vector of mu and sigma
        self.mu = np.zeros(len(self.models))
        self.sigma = np.zeros(len(self.models))

        for i in range(len(self.models)):
            self.mu[i] = np.mean(self.models[i].X_predict[self.validation_indices])
            self.sigma[i] = np.std(self.models[i].X_predict[self.validation_indices])


    def compute_model(self, B):
        #obtains a linear combination of the X predict matrix
        self.X_predict = np.zeros( self.X.shape)
        for i in range(len(self.models)): #maybe could be faster
            self.X_predict = self.X_predict + B[i] *(self.models[i].X_predict - self.mu[i])/self.sigma[i]

    def error(self):
        return super(SA, self).score(self.validation_indices)

    def compute_model_with_error(self, B):
        self.X_predict = np.zeros( self.X.shape)
        for i in range(len(self.models)): #maybe could be faster
            self.X_predict = self.X_predict + B[i] *(self.models[i].X_predict - self.mu[i])/self.sigma[i]
        return super(SA, self).score(self.validation_indices)

    def fit(self):
        self.normalize_values()
        #this is the actual part of the program
        #self.mu = np.zeros(len(self.models))
        #self.sigma = np.ones(len(self.models))
        #B =np.zeros(len(self.models))
        B = np.random.randn(len(self.models))
        error =self.compute_model_with_error(B)
        #print(self.X[self.validation_indices[:, 0], self.validation_indices[:, 1]])
        T = 100
        while(T > self.eps):
            B_prime = self.rand_modify(B)
            error_prime = self.compute_model_with_error(B_prime)
            value = np.exp((error - error_prime)/T)
            if( min( value, 1)  > np.random.uniform()):
                B = B_prime
                error = error_prime
            print(error)
            T = T* .99
        print(error)
        return B

    #rand_modify part
    def rand_modify(self, B):
        B_prime = B.copy()
        if( np.random.uniform() > .5):
            #THis is the rule where you choose two indices and change their values
            i, j = random.sample(set(range(len(self.models))), 2)
            x = min(B[i], B[j])*random.uniform(0, .02)
            B_prime[i] = B[i]  + x
            B_prime[j] = B[j] - x
        else:
            i = random.sample(set(range(len(self.models))), 1)
            B_prime[i] = B[i] * random.uniform(.8, 1.2)
        return B_prime








class regression_ensemble(recsys.recsys):

    def __init__(self,models, validation_indices, score_helper = None, eps = .001):
    #def __init__(self,models, train_indices, validation_indices, test_indices, score_helper = None, eps = .01):

        #indices are the actual indices of the matrix X
    #simulated annealing returns B
    #return B
        self.X = models[0].X
        self.score_helper = score_helper
        self.models = models
        #self.train_indices = train_indices
        self.validation_indices = validation_indices
        self.eps = eps

    def fit(self):
        #create a linear regression between the predictions in the validation dataset
        X_model = np.zeros((self.validation_indices.shape[0], len(self.models)))
        for i in range(len(self.models)):
            X_model[:, i] = self.models[i].X_predict[self.validation_indices[:, 0], self.validation_indices[:, 1]]
        y = self.X[self.validation_indices[:, 0], self.validation_indices[:, 1]]
        clf =  LinearRegression(False)
        clf.fit(X_model, y)
        self.X_predict = np.zeros( self.X.shape)
        self.coef= clf.coef_
        for i in range(len(self.models)): #maybe could be faster
            self.X_predict = self.X_predict + clf.coef_[i] *(self.models[i].X_predict) #Compute linear sum between models
        return clf.coef_


    def score(self, truth_index):
        return super(regression_ensemble,  self).score(truth_index)

    def predict_for_user(self, user_ratings, user_feat, k, feature_transform_all =None):
        self.predicted_values = np.zeros((1, self.X.shape[0]))
        for i in range(len(self.models)):
            self.models[i].predict_for_user(user_ratings, user_feat, k, feature_transform_all)
            self.predicted_values = self.predicted_values + self.models[i].predicted_values
        result = np.argsort(-1*self.predicted_values.T) #predict top results
        return result[0:k]



#
# #
# X = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 0]])
# models = list()
# for i in range(3):
#     models.append(pop_rec.pop_rec(X))
# models[0].X_predict = np.array([[.5, -.25, 1], [1, 1, .25], [0, 0, 0]])
# models[1].X_predict = np.array([[.7, .8, 1], [1, .9, .9], [0, 0, 0]])
# models[2].X_predict = np.array([[1, 1.5, 1], [1, .9, .2], [0, 0, 0]])
# validation_indices = np.array([[0, 0, 1], [0, 1, 2]]).T
#
# ensemble(models, SA)



# for i in range(3):
#     print(evaluate.rmse(X, models[i].X_predict, validation_indices) )
#
#
#
#
#
# #print(models)
# test = regression_ensemble(models, validation_indices, evaluate.rmse)
# test.run()