__author__ = 'John'
from mall_count_dataset import dict as data
import re
import numpy as np
from sklearn import decomposition
from numpy import linalg as LA

def get_category_matrix():
    #get the category count matrix from the joe jean dataset.
    #This dataset is clean
    #constants
    category_size = 0
    category_id = dict
    mall_id = dict
    mall_size = 0
    category_id = {'stupid': 100000000000} #fillers to define the type in the dictionary
    mall_id = {'stupid': 100000000000}
    #setting up category representation first
    for mall in data:
        for category in data[mall]:
            if not category in category_id:
                category_id[category] = category_size
                category_size = category_size+1
        mall_id[mall] = mall_size
        mall_size = mall_size +1


    for category in (' accessories', ' beauty products', ' clothing',):
        correct_name = re.sub('^ ', '', category)
        category_id[correct_name] = category_id[category]
        category_id.pop(category)


    category_id.pop('stupid')
    mall_id.pop('stupid')

    #convert the count dataset into a matrix
    X = np.zeros((category_size, mall_size))
    for mall in data:
        for category in data[mall]:
            correct_name = re.sub('^ ', '', category)
            X[category_id[correct_name], mall_id[mall]] = data[mall][category]

    #create category vector
    feature_names = {v: k for k, v in category_id.items()}
    return (X, feature_names)


def nmf_feature_extraction(X, n_topics=10, sparse_degree = 1, rand_id=40):
    #This is used to get features for malls based off their topics
    #doing nmf on the topics
    mall = decomposition.NMF(n_components=n_topics, sparseness='components', beta=sparse_degree, random_state = rand_id ).fit(X)
    #can access by fit_transform
    return(mall) #see if the reconstruction error matches for the same error.

def get_topics(X, feature_names, n_topics=10, n_top_words=10,  sparse_degree=1 ):
    #obtains the topics of nmf
    nmf = decomposition.NMF(n_components=n_topics, sparseness='components',  beta=sparse_degree ).fit(X.T) #l1 sparseness

    for topic_idx, topic in enumerate(nmf.components_):
        print( "Topic #%d:" % topic_idx)
        print( " ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

(X, feature_names) = get_category_matrix()
get_topics(X, feature_names, n_topics=50)