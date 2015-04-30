__author__ = 'John'
#modify csv files.

#lowercase every column.
#remove stop words
#remove states


#os.getcwd()
#os.chdir("/tmp/")
#os.getcwd()

import csv
import sys
import operator
import re
import os

def write_nice_csv(file_name):
    reader = csv.reader(open("csv/%s" % file_name), delimiter=",")


    #omit header
    next(reader, None)
    reader = sorted(reader, key=operator.itemgetter(0), reverse=False)
    sortedlist = sorted(reader, key=operator.itemgetter(1), reverse=False)

    #remove any rows with these states
    bad_states = {'alaska', 'hawaii', 'puerto rico'}

    #invoke writer and save the results in a nice_csv
    writer = csv.writer(open("csv/nice_csv/%s" % file_name, 'w', newline=''))
    for row in sortedlist:

        #remove states
        row[0] = row[0].lower();
        row[1] = row[1].lower();
        if row[1] not in bad_states:
            #remove stop words
            row[0] = re.sub("county|city|area|municipio|municipality| ", "", row[0]); #filtering a stop word
            writer.writerow(row);

def clean_loc():
    file_name = "zip_codes_states.csv";
    reader = csv.reader(open("join/%s" % file_name), delimiter=",")

    #open state reader
    state_file = "states.csv";
    state_reader = csv.reader(open("join/%s" % state_file), delimiter=",")
    state_dict = {};
    #set dictionaries
    for row in state_reader:
        #remove states
        state_dict[row[2]] = row[1];


    #omit header
    next(reader, None)
    #hi = list(reader)
    #print(hi)
    #print(operator.itemgetter(5))
    #reader = sorted(reader, key=operator.itemgetter(5), reverse=False)
    #sortedlist = sorted(reader, key=operator.itemgetter(4), reverse=False)
    sortedlist = list(reader);
    #remove any rows with these states
    bad_states = {'PR', 'AL', 'HI', 'VI', 'AE', 'AA', 'AP', 'AS'}

    #invoke writer and save the results in a nice_csv
    writer = csv.writer(open("join/new_%s" % file_name, 'w', newline=''))
    for row in sortedlist:

        #remove states
        row[3] = row[3].lower();
        row[5] = row[5].lower();
        if row[4] not in bad_states:
            #remove stop words
            row[5] = re.sub("county|city|area|municipio|municipality| ", "", row[5]); #filtering a stop word
            if(row[4] in state_dict):
                row[4] = state_dict[row[4]];
                writer.writerow(row);



#creates the dictionary to translate from abbreviated to full and translate fips dataset into this form.
def abbr_to_full():
    #updates states.csv
    state_file = "states.csv";
    reader = csv.reader(open("join/%s" % state_file), delimiter=",")
    state_dict = {};
    #set dictionaries
    for row in reader:
        #remove states
        state_dict[row[2]] = row[1];


    return state_dict;




#Determine the full form of the states name in the fips dataset.
def full_fips(): #makes fips into cleaner form
    #row[0] represents the abbreviated states
    state_dict = abbr_to_full();
    filename = "fips.csv";
    reader = csv.reader(open("FIPS and Population/%s" % filename), delimiter=",")
    next(reader, None)
    #sort the fips data set

    reader = sorted(reader, key=operator.itemgetter(3), reverse=False)
    sortedlist = list(reader);
    #remove any rows with these states
    #bad_states = {'PR', 'AL', 'HI', 'VI', 'AE', 'AA', 'AP', 'AS'}
    bad_states = {'PR', 'AK', 'HI'}

    #invoke writer and save the results in a nice_csv

    writer = csv.writer(open("join/%s" % filename, 'w', newline=''))
    for row in sortedlist:
        #row[3] = row[3].lower();
        if row[0] not in bad_states:
            #remove stop words
            if(row[0] in state_dict):
                row[0] = state_dict[row[0]];
                #row[3] = re.sub("county|city|area|municipio|municipality| ", "", row[3]); #filtering a stop word
            writer.writerow(row);

def write_nice_csv2(file_name):
    reader = csv.reader(open("csv/%s" % file_name), delimiter=",")


    #omit header
    next(reader, None)
    reader = sorted(reader, key=operator.itemgetter(0), reverse=False)
    sortedlist = sorted(reader, key=operator.itemgetter(1), reverse=False)

    #remove any rows with these states
    bad_states = {'alaska', 'hawaii', 'puerto rico', 'Alaska', 'Hawaii', 'Puerto Rico'}

    #invoke writer and save the results in a nice_csv
    writer = csv.writer(open("csv/nice_csv/%s" % file_name, 'w', newline=''))
    for row in sortedlist:

        #remove states
        if row[1] not in bad_states:
            #remove stop words
            writer.writerow(row);


#have a list of nice directories
directory_loc ="csv/nice_csv";
fips_filename = "fips.csv";
result_filename = "demographics.csv"
directory_list = list();
reader = list();


for file in os.listdir(directory_loc):
    if file.endswith(".csv"):
        directory_list.append(file);
        reader.append(list(csv.reader(open("csv/nice_csv/%s" % file), delimiter=","))); #create a list of .csv reader objects.

#create new table.

reader_fips = csv.reader(open("join/%s" % fips_filename), delimiter=",")

fips_header = ["USPS",	"GEOID",	"ANSICODE",	"NAME",	"POP10",	"HU10",	"ALAND",	"AWATER",	"ALAND_SQMI",	"AWATER_SQMI",	"INTPTLAT",	"INTPTLONG"];
writer = csv.writer(open("join/%s" % result_filename, 'w', newline=''))
desired_elements = [3, 0, 1, 10, 11, 4, 5, 6, 7, 8, 9] #obtains the desired elements from the list

for i in range(len(directory_list)):
    directory_list[i]= re.sub("\.csv|[(),]", "", directory_list[i] )
    directory_list[i]= re.sub(" ", "_", directory_list[i] )
    directory_list[i] = directory_list[i].lower();

header =  list(operator.itemgetter(*desired_elements)(fips_header));
header =header + directory_list;
writer.writerow(header);

j = 0;
for row in reader_fips:
    result_row = list(operator.itemgetter(*desired_elements)(row));
    for i in range(0, len(reader)):
        result_row.append(reader[i][j][2]);
    j=j+1;
    #write the columns based on the files that were read.
    writer.writerow(result_row);
#     #print(type(row));
#     #print(type(result_row));


































#clean_loc();
#prints out all the files and converts them into nice files.
# for file in os.listdir("csv"):
#     if file.endswith(".csv"):
#         write_nice_csv(file);



# #verifies if each list is the same
# #determine if two lists are equivalent
# file_name2 = "Percent Male.csv";
# #Determines if the counties list for each list are identical
# for file in os.listdir("csv/nice_csv"):
#     if file.endswith(".csv"):
#         compare_counties("Percent Asian.csv", file);


#This part edits the lines of the table



#
# #clean_loc();
# state_file = "states.csv";
# reader = csv.reader(open("join/%s" % state_file), delimiter=",")
# state_dict = {};
# #set dictionaries
# for row in reader:
#     #remove states
#     state_dict[row[2]] = row[1];

# #location
# #county - row[5]
# #state - row[4]
#
# #convert data into new_state.csv into nice format.
#
# #read each county from a demographic .csv file, then find the correspond longitude and latitude for that
#
# #demo
# #county - row[0]
# #state - row[1]
# location_file = "new_zip_codes_states.csv";
#
#
# demo_file = "Median Age.csv";
# demo_reader = csv.reader(open("join/%s" % demo_file), delimiter=",")
# writer = csv.writer(open("join/updated_%s" % demo_file, 'w', newline=''))
#
# for county in demo_reader:
#     loc_reader = csv.reader(open("join/%s" % location_file), delimiter=",")
#     for location in loc_reader:
#
#         if(location[4] == county[1] and location[5] == county[0]):
#             row = county;
#             row.append(location[1]);
#             row.append(location[2]);
#             writer.writerow(row);
#             break;
