import csv
from nmf_analysis import get_category_matrix
import os
import re
import pandas
#dir = "C:\\Users\\John\\Dropbox\\NYU COURSES\\Spring 2015\\Real Time Analytics and Big Data\\find_best_mall\\recomendation system\\"
#file = dir + "\\final_mall_with_demo.csv"
#file = os.path.abspath('final_mall_demo_data.csv')
mall_dic = {}
#hi = open(file, 'rU')
#go through function again to get old id.


with open(os.path.abspath('final_mall_demo_data.csv'), 'rU') as f:
    reader = csv.reader(f, delimiter=",")
    title_row = next(reader)
    for row in reader:
        mall_name = row[1].lower()
        category = row[-1]  #get old rows
        category = re.sub(', ', ',', category)
        category = category.split(",")
        mall_dic[mall_name]={}
        if (category == [""]):
            continue
        for i in category:
            (key, val)= i.split(":")
            key = re.sub('^ ', '', key)
            mall_dic[mall_name][key] = int(val)


f.close()
df = pandas.DataFrame(mall_dic).T.fillna(0)
df.as_matrix()
category_list = list(df.columns.values)
#print(df.head())
#
# user_feat = get_category_matrix(mall_dic)