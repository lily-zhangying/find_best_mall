import csv

dir = "/Users/lily/workspace/crwaler/yelp_crawler/yelp_crawler/yelp_crawler/dataset/"
# file = dir + "final_store.csv"
chunksize = 5000
fid = 1

with open(dir + 'final_store.csv', 'rb') as csvfile:
    f = open(dir + 'final_store_%d.csv' %fid, 'w')
    writer = csv.writer(f, delimiter=',')
    reader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(reader):
        writer.writerows([row])
        if not i%chunksize:
            f.close()
            fid += 1
            f = open(dir + 'final_store_%d.csv' %fid, 'w')
            writer = csv.writer(f, delimiter=',')
    f.close()