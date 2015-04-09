# merge seperate files
import glob
import csv
import scipy

dir = "/Users/lily/workspace/find_best_mall/yelp_crawler/yelp_crawler/result"

file_count = len(glob.glob(dir + "/*.csv"))
# i=0;
# for file in glob.glob(dir + "/*.csv"):
#     # i=i+1;
#     with open(file, "rU") as f:
