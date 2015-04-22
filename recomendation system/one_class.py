__author__ = 'John'
import numpy as np;
import random
import math
import scipy.sparse.bsr
from sklearn.cross_validation import train_test_split, KFold
from numpy.linalg import inv
from sklearn.decomposition import ProjectedGradientNMF
from itertools import groupby
import itertools
import similarity
import cf;
import nmf_analysis
import content
import wlas
import evaluate
import pop_rec
from sklearn import cross_validation

class one_class:
    def __init__(self, filename = 'dummy.txt', learner =None, X=None):
        self.learner = learner; #X is the matrix that we are dealing with. The rows are items and columns are users.
        #training_data #binary mat determining which is enteries are in the training set
        #testing_data #binary mat determining which is enteries are in the testing set
        self.writing_string =""
        self.filename = filename
        self.file = open(filename, "w+")
        self.string_parameters = list()
        self.corresponding_values = list()

            #default constants
        #cons =

    #partitions data into training and testing data by percentage
    def cv(self, k):
        #output: gives you a list of items to output
        X = self.learner.X
        #find indices of ones and put them into training/testing sets
        ones_x, ones_y = np.nonzero(X[: ,:] == 1)
        one_coord =np.array([ones_x, ones_y]);
        one_coord = one_coord.T
        np.random.shuffle(one_coord)
        kf_ones = cross_validation.KFold(one_coord.shape[0], n_folds=k)

        #find indices of ones and put them into training/testing sets
        zero_x, zero_y = np.nonzero(X[: ,:] == 0)
        zero_coord = np.array([zero_x, zero_y]);
        zero_coord = zero_coord.T
        np.random.shuffle(zero_coord)
        kf_zeros = cross_validation.KFold(zero_coord.shape[0], n_folds=k)


        training = list()
        testing = list()

        for ones, zeros in zip(kf_ones, kf_zeros):
            training.append(np.concatenate((one_coord[ones[0]], zero_coord[zeros[0]]),axis=0))
            testing.append(np.concatenate((one_coord[ones[1]], zero_coord[zeros[1]]),axis=0))
            #This makes the training set



        #create a numpy array
        return (training, testing)
        #return( (np.concatenate((ones_train, zero_train),axis=0), np.concatenate((ones_test, zero_test),axis=0) ))
        #concatenate the training and test array
    #equal cv for each user

    def cv_parameter_tuning(self, k, learner_dict=None, fun_list = None):

        training_ind, testing_ind=self.cv(self.learner.X, k)
        self.values = list()
        self.corresponding_values = list()
        for test in testing_ind:
            self.string_parameters = list()
            self.recursive_parameter_tuning(self.learner, test, learner_dict =learner_dict, fun_list=fun_list)
            self.values.append(self.corresponding_values)
            self.corresponding_values = list()
            self.file.close()



    def train_test_split_percent(self, percent):
        X = self.learner.X
        #keywords:
        #percent - the percent that you want the traiining data to be random
        #folds - number of folds you are working with

        #find indices of ones and put them into training/testing sets
        ones_x, ones_y = np.nonzero(X[: ,:] == 1)
        one_coord =np.array([ones_x, ones_y]);
        one_coord = one_coord.T
        np.random.shuffle(one_coord)
        ones_train, ones_test = train_test_split(one_coord, test_size=percent)

        #find indices of ones and put them into training/testing sets
        zero_x, zero_y = np.nonzero(X[: ,:] == 0)
        zero_coord = np.array([zero_x, zero_y]);
        zero_coord = zero_coord.T
        np.random.shuffle(zero_coord)
        zero_train, zero_test = train_test_split(zero_coord, test_size=percent)

        #create a numpy array
        return( (np.concatenate((ones_train, zero_train),axis=0), np.concatenate((ones_test, zero_test),axis=0) ))
        #concatenate the training and test array
    #equal cv for each user

    #partitions data into training and testing data by percentage
    def train_test_split_equal_user(self, X, percent):
        #keywords:
        #percent - the percent that you want the traiining data to be random
        #folds - number of folds you are working with

        #find indices of ones and put them into training/testing sets
        #go through each user and randomly split
        for i in range(X.shape[1]):

            ones_x, = np.nonzero(X[: ,i] == 1)
            np.random.shuffle(ones_x) #fix this
            ones_x = ones_x.T
            ones_train, ones_test = train_test_split(ones_x, test_size=percent)

            #find indices of ones and put them into training/testing sets

            zero_x, = np.nonzero(X[: ,i] == 0)
            np.random.shuffle(zero_x)
            zero_x = zero_x.T
            zero_train, zero_test = train_test_split(zero_x, test_size=percent)

            #concatenating stuff
            train = np.concatenate((ones_train, zero_train),axis=0)
            test= np.concatenate((ones_test, zero_test),axis=0)
            train = np.column_stack((train, i*np.ones((train.shape[0], 1))))
            test = np.column_stack((test, i*np.ones((test.shape[0], 1))))
            if i == 0:
                result_train = train
                result_test = test
            else:
                result_train = np.concatenate((result_train, train),axis=0)
                result_test = np.concatenate((result_test, test),axis=0)

            #create a numpy array
        return( (result_train, result_test ))
            #concatenate the training and test array


    def function_plugger(self, fun,fun_dict):
        parameter_name = list()
        iterating_values = list()
        possible_parameters = list()
        for key, value in fun_dict.items(): #breaking dictionary into two lists
            parameter_name.append(key)
            iterating_values.append(value)
        enumerated_values = list(itertools.product(*iterating_values))
        possible_functions = list()
        for combo in enumerated_values:
            parameters = dict(zip(parameter_name, combo)) #this may be enumerated
            possible_parameters.append(parameters)
            possible_functions.append(fun(**parameters))
        return (possible_functions, possible_parameters)

    def recursive_parameter_tuning(self, learner, test_ind, learner_dict=None, fun_list=None):

        #fun_list is a tuple with (name of function, actual helper function, dictionary ('parameter string', domain)
        #output best value
        #best combo, a multidimensional dictionary with d[name of function][parameter] = value
        best_value = 0
        if(fun_list == None or len(fun_list) ==0 ):
            pass
        else:
            best_combo = dict()
            fun_list_copied = fun_list.copy()
            current_function = fun_list_copied.pop()
            name = current_function[0]
            (possible_functions, possible_combinations) = self.function_plugger(current_function[1], current_function[2])
            #go through two lists at the same time
            old_string = self.writing_string

            for fun, current_combo in zip(possible_functions, possible_combinations):
                learner.get_helper2(name , fun)
                self.writing_string =self.writing_string + name+str(current_combo) #This is for writing stuff
                (value, combined_combo)= self.recursive_parameter_tuning(learner, test_ind, learner_dict, fun_list_copied)
                self.writing_string = old_string
                if(value >best_value):
                    best_value= value
                    copy_combined_combo = combined_combo #may not be necessary to do this
                    copy_combined_combo[name] = current_combo
                    best_combo = copy_combined_combo
                    #best_combo[name] =  zip( current_function[2].keys(), current_combo)
            return (best_value, best_combo)



        if(learner_dict==None or len(learner_dict) ==0):
            #combined_combo is an empty dictionary
            combined_combo = dict()
            learner.fit(test_indices=test_ind)
            value = learner.score(test_ind)
            self.string_parameters.append(self.writing_string)
            self.corresponding_values.append(value)
            self.writing_string =self.writing_string+str(value)+'\n'
            self.file.write(self.writing_string)
            print(self.writing_string)
            return (value, combined_combo) #just run test here
        else:
            best_value =0
            best_parameters = dict()
            parameter_name = list()
            iterating_values = list()
            #possible_parameters = list()
            for key, value in learner_dict.items(): #breaking dictionary into two lists
                parameter_name.append(key)
                iterating_values.append(value)
            enumerated_values = list(itertools.product(*iterating_values))
            old_string = self.writing_string
            for combo in enumerated_values:
                parameters = dict(zip(parameter_name, combo)) #this may be enumerated
                #possible_parameters.append(parameters)
                learner.get_parameters_2(parameters) #awesome
                self.writing_string =self.writing_string + 'learner'+str(parameters) #This is for writing stuff
                (value, combined_combo)= self.recursive_parameter_tuning(learner, test_ind)
                self.writing_string = old_string
                if(value >best_value):
                    best_value= value
                    combined_combo['learner'] = combined_combo
                    best_parameters = combined_combo
            return (value, best_parameters)






#a 50 by 10 matrix

# X = np.concatenate((np.ones((25, 10)), np.zeros((25, 10)) ),axis=0)
# one_class(X)
# print("lol")
# dawg = one_class(X);
# dawg.cv_percent(X, .20)
# #recursive_parameter_tuning(self, learner, test_ind, learner_dict=None, fun_list=None)
#
# def stupid_fun1(a, b=1, c=1):
#     return lambda d: d*a*b *c
# def stupid_fun2(e, f=1, g=1):
#     return lambda d: d*e*f *g
#
# fun_list = [['similarity_helper', stupid_fun1, {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 5, 4]}], \
#             ['feature_helper', stupid_fun2, {'e':[1, 2, 3], 'f': [4, 5, 6], 'g': [1, 3, 2]}]]
#
# X = np.array([[1, 1,1, 1, 0 ], [1, 1, 0, 0, 0], [1, 0, 1, 0, 0]])



#
# X = np.array([[1, 1, 1, 1] , [1, 1, 0, 0], [1, 0, 1, 0]])
# user_feat = np.array([[1, 1, 1, 2, 3], [0, 0, 4, 5, 6], [1, 0, 7, 8, 9], [0,1 , 10, 11, 12]])
# item_feat = None
# fun = content.user_to_item_helper(2, 4)
#
# cosine = similarity.cosine()
# content_helper = wlas.content_based_weight(fun,cosine)
#
# learner_dict = {"n_topics": [1, 2], "sparseness": [1, 2, 3] }
#
#
#
#
# learner = wlas.wlas(X, score_helper=evaluate.rmse, feature_helper = content_helper, user_feat=user_feat, item_feat=item_feat, n_topics=2)
#
#
# test = one_class()
# train_ind, test_ind = test.cv_percent(X, .20)
# test.recursive_parameter_tuning(learner, test_ind, learner_dict=learner_dict)
# #3 users and 5 items
#
#
#




#
# helper_functions = {"feature":nmf_analysis.mall_latent_helper, "similar":similarity.gaussian}
# iter_consts = {"n_topics" : ([5, 10, 20, 50], "feature"), 'sparse_degree' :([.1, .2, .3, .5, 1, 1.2], "feature"), 'alpha' : ([.1, .2, .3, .4], "similar") }
# hi = one_class()
# X=np.zeros((2, 2))
# learner = cf.cf(X)
# hi.parameter_tuning(learner, iter_consts, helper_functions)
# #iter_consts = {"n_topics" : ([5, 10, 20, 50], "feature"), 'sparse_degree' :([.1, .2, .3, .5, 1, 1.2], "feature") }
