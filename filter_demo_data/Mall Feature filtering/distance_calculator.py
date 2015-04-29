from haversine import haversine
import itertools
import json
import csv



def mall_distance(m1, m2):
	mall1 = (float(m1['mall_latitude']), float(m1['mall_longitude']))
	mall2 = (float(m2['mall_latitude']), float(m2['mall_longitude']))
	return haversine(mall1, mall2, miles = True)

mallsdatafile = open("mallsdata.txt", "rb")
mallsdistancefile = open("mallsdistance.csv", "w")
writer = csv.writer(mallsdistancefile, dialect = "excel")

#Header Row
writer.writerow(["From "]+["To "]+["Distance (in miles)"])

mallsdata = json.load(mallsdatafile)

for pair in itertools.combinations(mallsdata, 2):
	print("Distance between "+ pair[0]["mall_name"]+ " and " + pair[1]["mall_name"]+ " : " + str(mall_distance(pair[0],pair[1])))
	writer.writerow([pair[0]["mall_name"]]+[pair[1]["mall_name"]]+[mall_distance(pair[0],pair[1])])
	
mallsdistancefile.close()
mallsdatafile.close()
