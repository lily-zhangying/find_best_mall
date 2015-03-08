import csv
import glob

dir = "/Users/lily/PycharmProjects/filter/dataset/"

exist_unique_keys = {}
data = {}
counties = {}

for file in glob.glob(dir + "*.csv"):
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        title_row = next(reader)
        for row in reader:
            area_fips = row[0]
            own_code = row[1]
            industry_code = row[2]
            qtr = row[6]
            unique_key = industry_code + "_" + own_code + "_" + qtr
            if counties.has_key(area_fips) :
                if unique_key in counties[area_fips]:
                    print 'replica'
                else:
                    counties[area_fips].append(unique_key)
            else:
                counties[area_fips] = [unique_key]

            if exist_unique_keys.has_key(unique_key):
                data[unique_key] = data[unique_key].append(row)
            else:
                data[unique_key] = [row]
    csvfile.close()

counties_set = []
for k in counties:
    counties_set.append(set(counties[k]))
u = list(set.intersection(*counties_set))

with open(dir + 'final_industry_data.csv', 'wb') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(title_row)
    for final_unique_key in u:
        for line in data[final_unique_key]:
            writer.writerow(line)
file.close()



