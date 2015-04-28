__author__ = 'John'
import pandas as pd

unique_stores = pd.DataFrame.from_csv("../filter_store_data/dataset/__sorted_final_store.csv",  index_col=False)
hi = unique_stores.sort(["name"])
hi.to_csv("../filter_store_data/dataset/__sorted_final_store2.csv")
#hi = unique_stores.replace(to_replace="\s*$", value='', regex=True)
#hi = hi.replace(to_replace="^\s*", value='', regex=True)

#hi = unique_stores
#hi = unique_stores.apply(lambda x: x.str.replace('san fran', 'fuck'))
#print(hi.ix[9345:9355])
#look for san francscio soup