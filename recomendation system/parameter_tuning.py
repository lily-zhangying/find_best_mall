__author__ = 'John'
import one_class
import cf
import nmf_analysis
import evaluate
import numpy as np
import pandas as pd
import similarity
import wlas
import pop_rec
import content
from nmf_analysis import mall_latent_helper as nmf_helper
import filter_demo_data
np.random.seed(9001)

result_directory = "Cross Validation Results/"
print("load Matrix...")
X, category = filter_demo_data.get_X()
X = X[0:100, 0:50]
category = category[0:50, :]
possible_topics = list(np.arange(4, 42, 1))
possible_sparseness = list(np.arange(.1, 3, .1))
possible_delta = list(np.arange(.1, 1, .1))
k_fold_validation =2

#setting up helpers
cosine = similarity.cosine()
similarity_helper = cosine
score_helper = evaluate.negative_rmse


print("Setting up Tuning...")

##############################User Based Recommendations##########################
####NMF#####
#This is the first thing to tune which is nonnegative matrix factorization

# nmf_cf = cf.cf(X, similarity_helper=similarity_helper, user_feat=category, score_helper=score_helper)
# possible_sparseness = list(np.arange(.1, 3, .4))
# fun_list = [['feature_helper', nmf_helper, {'n_topics': possible_topics, 'sparse_degree':possible_sparseness }]]
#
# initializer = one_class.one_class(learner=nmf_cf)
# train, test = initializer.train_test_split_percent(.20)
#
# #initializer.cv_parameter_tuning_on_validation(k_fold_validation, train, test, result_directory +"nmf_cf.csv", fun_list=fun_list)
# print("Completed Nonnegative Matrix Factorization :)")



####Geographic Data data#####
#INTPTLAT	INTPTLONG

slave = pd.read_csv("Demographic Filtering/mall_with_demographic_category.csv", encoding = "ISO-8859-1", index_col=False)
# slave = slave.ix[:, "INTPTLAT":"INTPTLONG"].as_matrix()
#
# #Distance will be hamming distance
# distance_cf = cf.cf(X, similarity_helper=similarity.haversine_helper, user_feat=slave, score_helper=score_helper)
# stuff = one_class.one_class(learner=distance_cf)
# stuff.cv_parameter_tuning_on_validation(k_fold_validation, train, test, result_directory +"distance_cf.csv")


####Demographic data#####
slave = slave.ix[:, "POP10":"Rental_vacancy_rate_percent"].as_matrix()
#Normalize the results by centering the mean to 0 and the the std to 1 for each column
features = np.divide( np.subtract(slave,  np.mean(slave, axis=0)), np.std(slave, axis=0))

#Distance will be hamming distance
demographic_cf = cf.cf(X, similarity_helper=similarity.haversine_helper, user_feat=features, score_helper=score_helper)
stuff = one_class.one_class(learner=demographic_cf)
stuff.cv_parameter_tuning_on_validation(k_fold_validation, train, test, result_directory +"distance_cf.csv")


####Industrial data#####

##############################Content Based Recommendations##########################
######distances
content.user_to_item_helper()



######geographic data


##############################WLAS##########################


# #Get user_based wlas
# learn_dict ={'n_topics': possible_topics, 'sparseness':possible_sparseness }
# user_wlas = wlas.wlas(X, score_helper=evaluate.rmse, feature_helper=wlas.user_weight)
# stuff = one_class.one_class(learner=user_wlas)
# stuff.cv_parameter_tuning_on_validation(k_fold_validation, train, test,filename=result_directory +"user_weight_wlas.csv", learner_dict= learn_dict)
#
#
# #get item_based
# learn_dict ={'n_topics': possible_topics, 'sparseness':possible_sparseness }
# item_wlas = wlas.wlas(X, score_helper=evaluate.rmse, feature_helper=wlas.item_weight)
# stuff = one_class.one_class(learner=item_wlas)
# stuff.cv_parameter_tuning_on_validation(k_fold_validation, train, test,filename=result_directory +"item_weight_wlas.csv", learner_dict= learn_dict)


#get
# learn_dict ={'n_topics': possible_topics, 'sparseness':possible_sparseness }
# fun_list = [['feature_helper',wlas.item_weight, {'delta': possible_delta }]]
# uniform_wlas = wlas.wlas(X, score_helper=evaluate.rmse, feature_helper=wlas.item_weight)
# stuff = one_class.one_class(learner=uniform_wlas)
# stuff.cv_parameter_tuning_on_validation(k_fold_validation, train, test,filename=result_directory +"uniform_weight_wlas.csv", learner_dict= learn_dict, fun_list=fun_list)


##############################POP Recommendation##########################
# cosine = similarity.cosine()
# possible_topics = list(np.arange(4, 24, 5))
# possible_sparseness = list(np.arange(.1, 3, .4))
# filename = "poprec_user_weight_results.txt"
# pop_recommender = pop_rec.pop_rec(X, score_helper=evaluate.rmse)
# stuff = one_class.one_class(filename=filename, learner=pop_recommender )
#
#
# train_ind, test_ind =stuff.train_test_split_percent(.20)
# result = stuff.recursive_parameter_tuning(pop_recommender , test_ind)
# print(result)
#
#
