import numpy as np



class recsys(object):
    #X is the truth
    def __init__(self,X):
        self.X = X
        self.X_predict = None
        self.X_train = None
        pass
    #get the necessary helper functions to do an analysis. may require more parameters for derived classes
    def get_helpers(self, feature_func = None, similarity_func = None):
        if ( not(feature_func == None) or (self.feature_helper == None)):
            self.feature_helper = feature_func;
        if ( not(similarity_func == None) or (self.similarity_helper== None)):
            self.similarity_helper = similarity_func;

    def get_parameters(self, **kwargs):
        pass
        #this varies from learner to learner. Some learners do not have this because they do not need to get learned


    def transform_training(self, train_indices):
        #train_incides must be a |Train_Data|-by-2 matrix.
        #train_indices come in tuples
        if(not isinstance(train_indices, np.ndarray)):
            raise Exception("Dawg, your training indices have to be an ndarray")
        self.X_train = self.X;
        #whoops this is wrong
        self.X_train[train_indices[:, 0], train_indices[:, 1]]  = np.zeros((1, train_indices.shape[0]))

    def fit(self, train_indices = "None", test_indices = "None"):
        pass
        #the code of the actual
        #i

    #in reality, this would not be used alot
    def predict(self, indices):
        if(not isinstance(indices, np.ndarray)):
            raise Exception("Dawg, your indices have to be an ndarray")
        return self.X_predict(indices[:, 0], indices[:, 1])

    def score(self, truth_index):
        if(not isinstance(truth_index, np.ndarray)):
            raise Exception("Dawg, your testing indices have to be an ndarray")
        return self.score_helper(self.X, self.X_predict, truth_index)
        #do ranked precision
        #first



