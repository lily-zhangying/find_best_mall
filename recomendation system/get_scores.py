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
import ensemble
np.random.seed(9001)

result_directory = "Cross Validation Results/"
print("load Matrix...")
X, category = filter_demo_data.get_X()

X = X[:10000, :]
category= category[:, :]

k =10
slave = pd.read_csv("Demographic Filtering/mall_with_demographic_category.csv", encoding = "ISO-8859-1", index_col=False) #intializing values

#setting up helpers
cosine = similarity.cosine()
similarity_helper = cosine
score_helper = evaluate.map
scorers = [score_helper, evaluate.rmse]


##############################User Based Recommendations##########################
####NMF#####
#This is the first thing to tune which is nonnegative matrix factorization

nmf_cf = cf.cf(X, similarity_helper=similarity_helper, user_feat=category, feature_helper= nmf_helper(), score_helper=score_helper)
#nmf_cf =  pop_rec.pop_rec(X, score_helper=score_helper)
initializer = one_class.one_class(learner=nmf_cf)
train, test = initializer.train_test_split_percent(.20)
training_ind, validation_ind=initializer.split_training(k, train)

models = list()
models_name = list()
models.append(nmf_cf)
models_name = ["nmf_cf" ]

pop_recommender = pop_rec.pop_rec(X, score_helper=score_helper)
models.append(pop_recommender)
models_name.append("pop_rec")

slave = pd.read_csv("Demographic Filtering/mall_with_demographic_category.csv", encoding = "ISO-8859-1", index_col=False)
slave = slave.ix[:, "INTPTLAT":"INTPTLONG"].as_matrix()
distance_cf = cf.cf(X, similarity_helper=similarity.haversine_helper(), user_feat=slave, score_helper=score_helper)
models.append(distance_cf)
models_name.append("distance_cf")

slave = pd.read_csv("Demographic Filtering/mall_with_demographic_category.csv", encoding = "ISO-8859-1", index_col=False)
slave = slave.ix[:, "Homeowner_vacancy_rate_percent":"Rental_vacancy_rate_percent"].as_matrix()
#eatures = np.divide( np.subtract(slave,  np.mean(slave, axis=0)), np.std(slave, axis=0))
demographic_cf = cf.cf(X, similarity_helper=similarity_helper, user_feat=slave, score_helper=score_helper)
models.append(demographic_cf)
models_name.append("percent_demographic_cf")

# slave = pd.read_csv("Demographic Filtering/mall_with_demographic_category.csv", encoding = "ISO-8859-1", index_col=False)
# slave = slave.ix[:, "Homeowner_vacancy_rate_percent":"Rental_vacancy_rate_percent"].as_matrix()
# features = slave
# fun = content.user_to_item_helper(0, features.shape[1]-1) #might have to do minus
# demographic_content = content.content(X, similarity_helper=similarity_helper, user_feat=features, score_helper=score_helper, feature_helper=fun)
# models.append(demographic_content)
# models_name.append("demographic_content")
#
# slave = pd.read_csv("Demographic Filtering/mall_with_demographic_category.csv", encoding = "ISO-8859-1", index_col=False)
# slave = slave.ix[:, "INTPTLAT":"INTPTLONG"].as_matrix()
# features = slave
# fun = content.user_to_item_helper(0, features.shape[1]-1) #might have to do minus
# geographic_content = content.content(X, similarity_helper=similarity.haversine_helper(), user_feat=features, score_helper=score_helper, feature_helper=fun)
# models.append(geographic_content)
# models_name.append("geographic_content")


# user_wlas = wlas.wlas(X, score_helper=score_helper, feature_helper=wlas.user_weight, n_topics = 25)
# models.append(user_wlas)
# models_name.append("user_wlas")
#
# item_wlas = wlas.wlas(X, score_helper=score_helper, feature_helper=wlas.item_weight, n_topics = 25)
# models.append(item_wlas)
# models_name.append("item_wlas")






# stuff.cv_parameter_tuning_on_validation(k_fold_validation, train, test,filename=result_directory +"pop_recommender.csv")

#save everything to a csv file
# scores = np.zeros((k, len(scorers)))
#
# for learner, learner_name in zip(models, models_name):
#     i = 0
#     print(learner_name)
#     for validate in validation_ind:
#
#         indices = np.concatenate((test, validate), axis=0)
#         learner.fit(test_indices=indices )
#         j = 0
#         for scorer in scorers:
#             learner.score_helper = scorer
#             scores[i, j] = learner.score(validate)
#             print("%s, %s: %s"% (i, j, scores[i, j]))
#             j = j+1
#
#         i = i +1
#     np.savetxt("Cross Validation Results/%s.csv" % learner_name, scores)
#     np.savetxt("Cross Validation Results/%s_average.csv" % learner_name, np.average(scores, axis=0))



scores = np.zeros((k, len(scorers)))
i =0
coefficients = np.zeros((k,len(models) ))
for validate in validation_ind:
    for k in range(len(models)):
        indices = np.concatenate((test, validate), axis=0)
        print("%s  Trained" %models_name[k])
        models[k].fit(test_indices=indices )

    combiner = ensemble.regression_ensemble(models, validate)
    combiner.fit( )
    j = 0
    for scorer in scorers:
        combiner.score_helper = scorer
        scores[i, j] = combiner.score(test)
        print("%s, %s: %s"% (i, j, scores[i, j]))
        j = j+1

    i = i +1
np.savetxt("Cross Validation Results/ensembler.csv", scores)
np.savetxt("Cross Validation Results/ensembler_average.csv" , np.average(scores, axis=0))


