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
class one_class:
    def __init__(self, data =None, item_mat = None, mall_mat = None):
        X = data; #X is the matrix that we are dealing with. The rows are items and columns are users.
        #training_data #binary mat determining which is enteries are in the training set
        #testing_data #binary mat determining which is enteries are in the testing set
        test = {0:None, 1:None, "data":None}
        train = {0:None, 1:None, "data":None}
        self.mall_mat = mall_mat
        self.item_mat = item_mat
            #default constants
        #cons =

    #partitions data into training and testing data by percentage
    def cv_percent(self, X, percent):
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
    def cv_equal_user(self, X, percent):
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



    def parameter_tuning(self, learner,  iter_consts, helper_functions, filename = "parameter_tuning_result.txt", testing_ind = None):
        #note the the leaner and score_helper must already be constructed
        #iter_consts is a dictionary:
            #key = name of the parameter
            #value = tuple (domain of the parameters it belongs into[], string indicating the function it tunes)
        #helper_functions is a dictionary:
            #key = category of the function
            #value = the actual function

        #This model assumes that iter_consts.value[1], helper_function.key, and one of the formal parameters in get_helpers will match
        parameter_category = list()
        parameter_name = list()
        iter_parameters = list() #goes through every combination of parameters and plug them into the formula
        multidimensional = list() #master list of the parameters
        file = open(filename, "w+")
        max_score = 0


        #iterating through parameter category
        for (key, value) in iter_consts.items():
            parameter_category.append(value[1])
            multidimensional.append(value[0])
            parameter_name.append(key)

        #creating multidimensional array enumerate all the different possibilities
        for element in itertools.product(*multidimensional):
            print(element)
            iter_parameters.append(element)

        for parameters in iter_parameters:
            #creating these parameters to put in model
            #creates the initial dictionary for each list.
            #key = category of the function
            #value = the actual function
            c = dict()
            #finds the category for the parameters
            for i in range(len(parameter_name)):
                #adds parameters to their respective category dictionary
                #adds parameters to function i if it matches
                if(parameter_category[i] in c):
                    c[parameter_category [i]][parameter_name[i]] = parameters[i]
                else:
                    c[parameter_category [i]] = {parameter_name[i] : parameters[i]}


            #plug in functions to use model and create algorithm
            #use add_function to the learner to plug in values if the function is not None

            #use add_functions by placing the functions into dictionaries
            #new diction
                #key = category name
                #value = function after being plugged in
            for category in list(set(parameter_category)): #goes through a unique list of parameter categories
                if( not(helper_functions[category] == None)):
                    if(len(c[category]) > 0):
                        #plugging in stuff for the function. It should work
                        f = helper_functions[category](**c[category])
                        learner.get_helpers(**{category : f}) #This needs to be tested
                    else:
                        f = helper_functions[category]()
                        learner.get_helpers(**{category: f})
                if(category == "learner"):
                    if(len(c[category]) > 0):
                        learner.get_parameters(**category)
            #This part tests the accuracy of the model after getting everything in placed
            learner.fit(test_indices = testing_ind)
            val = learner.score(testing_ind)
            #writing into text file
            key_input = "(" + ", ".join([str(x) for x in parameters] ) +")"
            file.write( "%s %s" % (key_input, val) )
            #create lines that will indicate that the the parameter tuned functions will become none
            learner.remove_helpers(list(set(parameter_category)))
            if(val > max_score):
                max_score = val
        return max_score








def bipartite_projection(adj, item_allocation):
    #gives you
    #adj is assumed to be the binary item-user matrix where item are the rows and user are the column
    #see paper in bipartite network projection and personal recommendation for reference
    adj = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1]]);
    size = adj.shape;
    item_deg=np.dot(adj, np.ones((size[1], 1))) #this finds the degree of the nodes in the item set
    item_power = np.transpose(adj * (1/item_deg));
    user_deg = np.dot(np.ones((1, size[0])),adj);
    user_power = adj * (1/user_deg)
    W = np.dot(user_power, item_power) #This tells you how much weight that item i will give to j depending how similar it is to j
    recommend_items= np.dot(W, item_allocation); #This will give you the item distribution for a certain users
    return (recommend_items, W);

#a 50 by 10 matrix

X = np.concatenate((np.ones((25, 10)), np.zeros((25, 10)) ),axis=0)
one_class(X)
print("lol")
dawg = one_class(X);
dawg.cv_percent(X, .20)







#
# helper_functions = {"feature":nmf_analysis.mall_latent_helper, "similar":similarity.gaussian}
# iter_consts = {"n_topics" : ([5, 10, 20, 50], "feature"), 'sparse_degree' :([.1, .2, .3, .5, 1, 1.2], "feature"), 'alpha' : ([.1, .2, .3, .4], "similar") }
# hi = one_class()
# X=np.zeros((2, 2))
# learner = cf.cf(X)
# hi.parameter_tuning(learner, iter_consts, helper_functions)
# #iter_consts = {"n_topics" : ([5, 10, 20, 50], "feature"), 'sparse_degree' :([.1, .2, .3, .5, 1, 1.2], "feature") }
