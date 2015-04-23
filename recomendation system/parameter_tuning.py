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
from nmf_analysis import mall_latent_helper as nmf_helper
import filter_demo_data

print("load Matrix...")
X, category = filter_demo_data.get_X()



print("Setting up Tuning...")

cosine = similarity.cosine()
nmf_cf = cf.cf(X, similarity_helper=cosine, user_feat=category, score_helper=evaluate.rmse)
possible_topics = list(np.arange(4, 42, 5))
possible_sparseness = list(np.arange(.1, 3, .4))
fun_list = [['feature_helper', nmf_helper, {'n_topics': possible_topics, 'sparse_degree':possible_sparseness }]]
filename = "nmf_cf.csv"
stuff = one_class.one_class(filename=filename, learner=nmf_cf)
train, test = stuff.train_test_split_percent(.20)
stuff.cv_parameter_tuning_on_validation(2, train, test, filename=filename, fun_list=fun_list)



# cosine = similarity.cosine()
# #possible_topics = list(np.arange(4, 42, 5))
# #possible_sparseness = list(np.arange(.1, 3, .4))
# possible_sparseness = [1, .5]
# #fun_list = [['feature_helper', nmf_helper, {'n_topics': possible_topics, 'sparse_degree':possible_sparseness }]]
#
# filename = "dummy.csv"
#
# #learn_dict = {'n_topics': possible_topics, 'sparseness':possible_sparseness }
# learn_dict = { 'sparseness':possible_sparseness }
# user_wlas = wlas.wlas(X, score_helper=evaluate.rmse, feature_helper=wlas.user_weight)
# stuff = one_class.one_class(filename=filename, learner=user_wlas)
# train, test = stuff.train_test_split_percent(.20)
# stuff.cv_parameter_tuning_on_validation(2, train, test, filename=filename, learner_dict= learn_dict)


# cosine = similarity.cosine()
# possible_topics = list(np.arange(4, 24, 5))
# possible_sparseness = list(np.arange(.1, 3, .4))
# filename = "poprec_user_weight_results.txt"
# user_wlas = wlas.wlas(X, score_helper=evaluate.rmse, feature_helper=wlas.user_weight)
# stuff = one_class.one_class(filename=filename, learner=user_wlas)
#
#
# train_ind, test_ind =stuff.train_test_split_percent(.20)
# result = stuff.recursive_parameter_tuning(user_wlas, test_ind)
# print(result)



#
#
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
