import csv
import glob
import os
import os.path
dir = "/Users/lily/workspace/find_best_mall/filter_industry_data/dataset/"
exist_unique_keys = {}
data = {}
counties = {}

file_count = len(glob.glob(dir + "2013.q1-q4.by_area/*.csv"))
i=0;
for file in glob.glob(dir + "2013.q1-q4.by_area/*.csv"):
    # print file
    i=i+1;
    with open(file, 'r') as csvfile:
        print("%i out of %i" % (i, file_count));
        reader = csv.reader(csvfile, delimiter=',')
        title_row = next(reader)
        for row in reader:
            area_fips = row[0]
            own_code = row[1]
            industry_code = row[2]
            qtr = row[6]
            unique_key = industry_code + "_" + own_code + "_" + qtr
            if area_fips in counties:
                if unique_key in counties[area_fips]:
                    print('replica')
                else:
                    counties[area_fips].append(unique_key)
            else:
                counties[area_fips] = [unique_key]

            if unique_key in data:
                # print 'append'
                data[unique_key].append(row)
            else:
                data[unique_key] = [row]
    csvfile.close()

counties_set = []
for k in counties:
    counties_set.append(set(counties[k]))
u = list(set.intersection(*counties_set))
with open(dir + 'final_industry_data.csv', 'wb') as file:
    writer = csv.writer(file, delimiter=',')
    #writer.writerow(title_row)
    for final_unique_key in u:
        for line in data[final_unique_key]:
            #print(line);
            #print(type(line))
            writer.writerow(line)
file.close()



