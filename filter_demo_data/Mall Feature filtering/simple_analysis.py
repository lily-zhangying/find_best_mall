from sklearn.neighbors.kde import KernelDensity
import numpy as np
import pandas;
from sklearn import cross_validation;
colnames = ['store_id', 'name', 'mall_id'];
file_name = 'store.csv'
store = pandas.read_csv('store.csv', names = colnames, encoding = "ISO-8859-1")
mall = pandas.read_csv('mall_with_demo.csv', header=0, encoding = "ISO-8859-1");





#cleaning columns
store.name = store.name.str.replace(" {2, }", "")
store.name = store.name.str.lower()





#select a particular store and obtain malls with this store
restriction = store.name.str.contains("uno chicago grill")
restriction.unique()
selected_stores = store.mall_id[restriction]

#crreate matrix for regression
regression_columns =np.arange(4, len(mall.columns))
X = mall[mall.mallid.isin(selected_stores)][regression_columns].values;
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
#run regression

#How do I test the accuracy of the model.