from sklearn.neighbors.kde import KernelDensity
import numpy as np
import pandas;
from sklearn import cross_validation;
from sklearn import linear_model;
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold


def logistic_regression(mall, selected_malls, name=None):
        #mall- original data
        #selected_malls - boolean vector indicating wheter a mall has a store.


    #create matrix and vector for regression    
    #the selected stores will be 1's and 0's otherwise for regression.
    regression_columns =np.arange(5, len(mall.columns))
    X = mall[:][regression_columns].values;
    y = mall.mallid.isin(selected_malls);
    y = y.astype(int); #get binary values for regression
    
    #L2- logistic regression
    logreg = linear_model.LogisticRegression();#L2- regression
    #logreg.fit(X, y);
    
    #compute validty of regression with cross validation
    scores = cross_validation.cross_val_score(logreg, X, y, cv=10)
    #the cross validation is not random
    
    #find which items are predicted incorrectly
    kf = KFold(len(X[:, 1]), n_folds=5)
    for train, test in kf:
        logreg.fit(X[train][:], y[train])
        wrong_samples = logreg.predict(X[test][:]) != y[test]
        print("Suppose to be: ")
        print(y[test[np.array(wrong_samples)]])
        
        
    print("The score for %s" % name);
    print(scores);
    return scores;
    
def kde(mall, selected_stores, name=None):
        #mall- original data
        #selected_stores - boolean vector indicating wheter a mall has a store.


    #create matrix and vector for regression for regression    
    #the selected stores will be 1's and 0's otherwise for regression.
    regression_columns =np.arange(5, len(mall.columns))
    X = mall[mall.mallid.isin(selected_stores)][regression_columns].values;   
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
    
    
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': np.linspace(0.1, 1.0, 30)},
                    cv=10) # 20-fold cross-validation
    grid.fit(X)
    kde = grid.best_estimator_
    #print(kde)
    return kde
#How do I test the accuracy of the model.


#obtain csv files
colnames = ['store_id', 'name', 'mall_id'];
file_name = 'store.csv'
store = pandas.read_csv('store.csv', names = colnames, encoding = "ISO-8859-1")
mall = pandas.read_csv('mall_with_demo.csv', header=0, delimiter="\t", encoding = "ISO-8859-1");

test_store = pandas.read_csv('test_store.csv', header=0);


#cleaning columns
#store.name = store.name.str.replace("\s*", " ")
store.name = store.name.str.replace(" {2, }", "")
store.name = store.name.str.lower()



#np.arange(5, len(mall.columns))
kde_results = dict();
log_results = dict();
for curr_store in test_store.name:
    #select a particular store and obtain malls with this store
    restriction = store.name.str.contains(curr_store)
    selected_malls = store.mall_id[restriction]
    log_results[curr_store] = logistic_regression(mall, selected_malls, curr_store);
    #kde_results[curr_store]= kde(mall, selected_malls)
