from haversine import haversine
import itertools
import json
import csv
import codecs

reader = codecs.getreader("utf-8")



def mall_distance(mall, county):
	mall = (float(mall['mall_latitude']), float(mall['mall_longitude']))
	county = (float(county['INTPTLAT']), float(county['INTPTLONG']))
	return haversine(mall, county, miles = True)

#Opening files
mallsdatafile = open("mallsdata.txt", "rb")
#countydatafile  = csv.DictReader(open('demographics.csv'))
mallsdistancefile = open("mall_with_county.csv", "w") #results
writer = csv.writer(mallsdistancefile, delimiter=",")
#Header Row
#writer.writerow(["Mall"]+["County"]+["Geoid"]+["Mall latitude"] +["Mall longitude"] + ["County longitude"] + ["County latitude"])
writer.writerow(["mallid"] +["mall"]+["state"] + ["county"]+["geoid"])

mallsdata = json.load(reader(mallsdatafile))

bad_states = ["Alaska", "Hawaii", "Columbia"];
i = 1;
for m in mallsdata:
    min_dist = float("inf"); #computes min distance
    nearest_county = ()
    #loads county data
    #this get rids of bad mall names
    something = m["mall_state"]
    if(m["mall_state"] not in  bad_states):
        county_file = open('demographics.csv')
        county  = csv.DictReader(county_file, delimiter=",")
        for c in county:
            dist = mall_distance(m,c);
            if(dist < min_dist):

                min_dist = dist;
                nearest_county = (c["NAME"], c["USPS"], c["GEOID"]); #This obtains important county information
            print("Distance between "+ m["mall_name"]+ " and " +c["NAME"]+ " in " + c["USPS"]+  ": " + str(dist));
        writer.writerow([i] + [m["mall_name"]]+[nearest_county[0]]+[nearest_county[1]]+[nearest_county[2]])
        county_file.close(); #reopens county file for each step.
    i=i+1;
	
mallsdistancefile.close()
mallsdatafile.close()
