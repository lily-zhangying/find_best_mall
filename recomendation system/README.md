## recommendation sysytem

* use several machine learning technologies to predict
	* Each file are classes of different machine learning technologies. Each of these classes have the functions: fit, predict, score and predict_for_user. Note that these functions are identical to the methods for the SK-learn machine learning modules.
	*For each of the recommendation classes, they have their own set of inputs. This can be inputted when you are calling the class. Typically, they require you to input the matrix X, your binary values of whether a store is a mall or not, and a feature matrix. Additionally, all the fit methods for these classes have the parameters (train_indices, test_indices), where train_indices and test_indices represent the training and test indices of the dataset. Note that both of these are not required values. If test_indices is given a matrix indicating the indices of that are in the testing set, then all the values where the indices indicate are turned to 0. In fit, it will create a X_predict matrix where tries to predict values based on algorithm of the recommendation system. Predict() has an input of the matrices consisting of indices of the matrix, where it will return the predicted values of X_predict based on the inputted matrix. Also, predict_for_user has the inputs (user_ratings, user_feat=None, k=10). user_ratings is a vector that indicates what stores does it have. user_feat maybe necessary since some of the algorithms require data about the user. k is the amount of items you recommend. 


### machine learning technologies
* item-based collaborative filtering
* user-based collaborative filtering
* top-k items
* content based recommendation
* occf
* combine all the results together

### Details

* File: filter_demo_data.py
    * get_X() is the function that enables you to get the binary matrix of the shops
    * get_final_demo_revisited() is used whenever one of the actual data files (csv) are updated

* File: mall_count_dataset.py
	* This is used to get the count entries of the stores for each mall that was collected by Joe Jean. It is in the form of a dictionary

* File: recsys.py
	* This is the general framework for all the recommendation systems for our dataset

* File: content.py
    * Content Based Recommendations

* File : cf.py
    * user-based collaborative filtering

* File : cf_item.py
    * item-based collaborative filtering

 * File : wlas.py (spelled incorrectly, suppose to be wals.py)
    * Weighted Least Aternating 

* File : pop_rec.py
    * top-k items recommendations

* File : logistic_reg.py
    * Used logistic regression as a recommendation system (Never used)

* File : similarity.py
    * Helper functions to compute the similarity between two objects
    
* File : evaluate.py
	* Scoring Functions that computes the validaty of a recommendation model. Specifically, it computes the MAP and MSE of the a model.

* File: command_line.py
	* This enables you to run a fast verision of our algorithm. This recommends stores to any arbitary mall based on User-Based Collaborative Filtering using county information as features.
