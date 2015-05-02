import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import linear_model #used for logistic regression
import filter_demo_data
import one_class
import recsys
import time


class logistic_reg(recsys.recsys):
    #baseline for recommendations
    #This will perform extremely badly with rmse as a score evaluator. but maybe not map
    def __init__(self,X, feature_helper = None, score_helper = None, user_feat = None, \
            sparseness = 1):
        self.X = X
        self.feature_helper = feature_helper
        self.score_helper = score_helper
        self.user_feat = user_feat
        self.sparseness = sparseness

        pass
    #get the necessary helper functions to do an analysis. may require more parameters for derived classes
    def get_helpers(self, feature_func = None, similarity_func = None):
        if ( not(feature_func is None) or (self.feature_helper is None)):
            self.feature_helper = feature_func;
        if ( not(similarity_func is None) or (self.similarity_helper is None)):
            self.similarity_helper = similarity_func;


    def predict_for_user(self, user_ratings, user_feat, k, feature_transform_all =None):
        #Will develop later.
        #save the coefficients of the linear regression into a matrix
        pass


    def transform_training(self, train_indices,  test_indices):
        #train_incides must be a |Train_Data|-by-2 matrix.
        #train_indices come in tuples
        self.X_train = np.copy(self.X);
        if((test_indices is None) and (train_indices is None) ):
            return
        elif(not (test_indices is None)):
            self.X_train[test_indices[:, 0], test_indices[:, 1]]  = np.zeros((1, test_indices.shape[0]))
            return
        else:
            #create a binary matrix that
            Nitems, Nusers = self.X.shape
            test_indicator = np.ones((Nitems, Nusers))

            test_indicator[train_indices[:, 0], train_indices[:, 1]]  = np.zeros((1, train_indices.shape[0]))
            self.X_train[test_indicator == 1] = 0



    def fit(self, train_indices = "None", test_indices = "None"):
        #super(logistic_reg, self).transform_training(train_indices, test_indices)
        t = time.time()
        self.X_train = self.X
        Nitems, Nusers = self.X_train.shape
        print(self.X_train.shape)
        print(self.user_feat.shape)
        #feature transformation
        if self.feature_helper == None:
            self.feature_transform = self.user_feat
        else:
            self.feature_transform = self.feature_helper(X=self.X_train, feat = self.user_feat)

        self.X_predict = np.zeros( (Nitems, Nusers))
        for i in np.arange(Nitems): #if this loop can be parallelized, that will be awesome :)
            #in the future, use vector binary classifier
            print(i)
            mall_in_training = train_indices[ train_indices[:, 0] == i, 1] #This gets the malls that are in training   for the ith store and makes a prediction off of that
            y_log = self.X_train[i, mall_in_training]
            X_log = self.feature_transform[mall_in_training, :]

            #L2 logistic regression
            logreg = linear_model.LogisticRegression(C=self.sparseness);#L2- regression
            #print(np.sum(y_log))

            logreg.fit(X_log, y_log)
            probability_scores = logreg.predict_proba(self.feature_transform)
            #verify probability scores for 0 and 1
            self.X_predict[i, :] = probability_scores[:, 1]

            #Save each predictor, logreg, in a list for regression latter.

            #logreg.coef_ #gives you the coefficent of the regression


        print( time.time() - t)
        return self.X_predict

    #in reality, this would not be used alot
    def predict(self, indices):
        if(not isinstance(indices, np.ndarray)):
            raise Exception("your indices have to be an ndarray")
        return self.X_predict(indices[:, 0], indices[:, 1])

    def score(self, truth_index):
        if(not isinstance(truth_index, np.ndarray)):
            raise Exception("your testing indices have to be an ndarray")
        return self.score_helper(self.X, self.X_predict, truth_index)
        #do ranked precision
        #first

    def get_helper2(self, name, function):
        if(name == 'feature_helper'):
            self.feature_helper = function
            return
        if(name == 'similarity_helper'):
            self.similarity_helper = function
            return
        if(name == 'score_helper'):
            self.score_helper = function
            return
        else:
            raise Exception("Cannot find feature function corresponding to the input name")


#for testing use category data
X, category = filter_demo_data.get_X()
print(X.shape)
print(category.shape)
X = X[:, :]
category = category[:, :]
model = logistic_reg(X, user_feat=category)

initializer = one_class.one_class(learner=model)
t = time.time()
train, test = initializer.train_test_split_equal_item(X, .1) #use something else. THe train test split gets ones sometimes
print(time.time() - t )
train = train.astype(int)
test = test.astype(int)
model.fit(train, test)
