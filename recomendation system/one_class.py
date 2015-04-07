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
np.random.seed(42);

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








    def parameter_tuning(self, learner,  iter_consts, helper_functions, filename = "parameter_tuning_result.txt", training_ind = None, testing_ind = None):
        #note the the leaner and score_helper must already be constructed
        #iter_consts is a dictionary:
            #key = name of the parameter
            #value = tuple (domain of the parameters it belongs into[], string indicating the function it tunes)
        #helper_functions is a dictionary:
            #key = category of the function
            #value = the actual function

        #This model assumes that iter_consts.value[1], helper_function.key, and one of the formal parameters in get_helpers will match


        parameter_category = list();
        parameter_name = list()
        iter_parameters = list()
        file = open(filename, "w+")
        max_score = 0


        #iterating through parameter category
        for (key, value) in iter_consts.items():
            parameter_category.append(value[1])
            parameter_name.append(key)

        #creating multidimensional array storing enumerate all the different possibilities
        for element in itertools.product(*iter_consts):
            print(element)
            iter_parameters.append(element)

        for parameters in iter_parameters:
            #creating these parameters to put in model
            #creates the initial dictionary for each list.
            c = dict()
            for i in range(len(parameter_name)):
            #key = category of the function
            #value = the actual function
                #adds parameters to their respective category dictionary
                #adds parameters to function i if it matches
                for (category, function) in helper_functions:
                    if(parameter_category [i] == category):
                        c[category] = {parameter_name[i] : parameters[i]}

            #plug in functions to use model and create algorithm
            #use add_function to the learner to plug in values if the function is not None

            #use add_functions by placing the functions into dictionaries
            #new diction
                #key = category name
                #value = function after being plugged in
            for category in parameter_category:
                if( not(helper_functions[category] == None)):
                    if(len(c[category]) > 0):

                        f = helper_functions[category](**c[category])
                        learner.get_helpers(**{category, f}) #This needs to be tested
                    else:
                        f = helper_functions[category]()
                        learner.get_helpers(**{category, f})
                if(category == "learner"):
                    if(len(c[category]) > 0):
                        learner.get_parameters(**category)
            #This part tests the accuracy of the model after getting everything in placed
            learner.fit(test_indices = testing_ind)
            val = learner.score(testing_ind)
            #writing into text file
            key_input = "(" + ", ".join([str(x) for x in parameters] ) +")"
            file.write( "%s %s" % (key_input, val) )

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


cons =  dict([('iter_max', 50), ('lambda', 0), ('error', 1)])
# cons["lambda"] = .10
# cons["error"] = 1

test = one_class()
tuple_size = (300, 30)
R = np.random.choice([0, 1], size=tuple_size, p=[.95, .05])
d = 10
model = ProjectedGradientNMF(n_components=d, init='random', random_state=0)
model.fit(R)

V_init = (model.components_).T #the matrix comes out as a d*n matrix, so you have to do the transpose to fix it.
#V_init = .1 * np.random.randn(tuple_size[1], d) + 0
W = np.ones(tuple_size)
test.wlas(R, W, d, V_init, cons)
