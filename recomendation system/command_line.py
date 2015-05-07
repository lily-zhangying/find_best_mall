__author__ = 'John'

import pandas as pd
import numpy as np
import re
import cf
import similarity
import filter_demo_data
cosine = similarity.cosine()
similarity_helper = cosine
import sys

def read_input(filename, top_N=10): #just reads in X and category matrix so loading it will not take time
    shop_mall = pd.read_csv("Demographic Filtering/mall_store_list.csv", encoding = "ISO-8859-1", index_col=False) #This is used for obttaining unique malls

    stores_db = shop_mall[["store", "store_id"]].drop_duplicates(subset = ["store"]) #get unique stores
    stores_db.index = pd.Series(np.arange(stores_db.shape[0]))
    #stores_db.to_csv("command_line_files/store_list.csv", header="true")

    mall_demographic = pd.read_csv("Demographic Filtering/mall_with_demographic_category.csv", encoding = "ISO-8859-1", index_col=False)
    county_db = mall_demographic[["county", "usps"]].copy(deep=True) #reads all
    county_db["county"] =county_db["county"].str.lower()
    county_db["usps"] =county_db["usps"].str.lower()
    county_db =county_db.drop_duplicates(["county", "usps"])
    #county_db.to_csv("command_line_files/county_list.csv", header="true")
    #save columns into another file

    #read txt file
    file = open(filename)
    entire_file = file.read()
    user = re.split('\n+', entire_file) #first entry of user is their county followed by stores
    #accept two type of inputs. One read
    X, _ = filter_demo_data.get_X()

    #get category features
    #This will enable you to get one user
    #get demographic features
    user_demographic_feature = mall_demographic.ix[mall_demographic["county"].str.lower() == user[0].lower(), "Homeowner_vacancy_rate_percent":"Rental_vacancy_rate_percent"].ix[mall_demographic["usps"].str.lower() == user[1].lower()].as_matrix()[0, :]

    #create Nstore vctor that represents the boolean values of a mall having a store.
    user_store_db = pd.DataFrame(pd.Series(user[2:]).str.lower(), columns=["store"] ) #put stuff into dataframe
    user_shop_index = pd.merge(user_store_db, stores_db, how="left", left_on=["store"], right_on=["store"] )["store_id"].as_matrix()
    user_ratings = np.zeros(stores_db.shape[0])
    user_ratings[user_shop_index] = 1
    user_ratings.reshape((-1, 1))

    #computing category features
    #get category features. You can do this by computing the weighted average of all the malls based on cosine distance

    #This is used to compute
    user_feature = mall_demographic.ix[:, "Homeowner_vacancy_rate_percent":"Rental_vacancy_rate_percent"].as_matrix()
    recommendation_system = cf.cf(X, similarity_helper=similarity_helper)
    top_recommendations = recommendation_system.predict_for_user(user_ratings, user_demographic_feature , top_N, user_feature ) #do another join
    print(stores_db.ix[top_recommendations, "store"] )

if  len(sys.argv) == 2:
    read_input(sys.argv[1])
elif len(sys.argv) == 3:
    read_input(sys.argv[1], sys.argv[2])