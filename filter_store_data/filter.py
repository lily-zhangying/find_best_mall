# filter the store data
# remove the redundant store's name
import csv
import re
import unicodedata
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

dir = "/Users/lily/workspace/find_best_mall/filter_store_data/dataset"
file = dir + "/store.csv"
final_file = dir + "/final_store.csv"
stores_dic = {}
store_name_frequency = {}

def remove_accent_marks(input_str):
    nkfd_form = unicodedata.normalize('NFKD', unicode(input_str))
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])

with open(file, 'rU') as store_file:
    reader = csv.reader(store_file, delimiter=",")
    for row in reader:
        # lowercase and trim string
        name = row[1].lower().strip()
        # replace several spaces to one space
        name = re.sub("\s+", " ", name)
        # remove special characters
        name = re.sub("(\s*)[&|'|\\\|\/](\s*)", "", name)
        #replace accent_marks of name
        name = remove_accent_marks(name)
        #filter atm & vending machines
        if (re.search("(\s*atm\s*)|(\s*vending\s*machines\s*)|(^(advanced)\s*)", name)):
            continue
        #filter open EXCEPT ["open advanced", "mri open mobile"]
        if (re.search("(^(open)\s*((?!(mobile))|(?!(advanced mri)))$)", name)):
            continue

        #remove [".co" ",the" "outet" "and co." "Now Open" "Opening"]
        name = re.sub("(\s*and\s*co\.\s*)|(\s*co\.\s*)|(\s*\,\s*the\s*)|(\s*outlet\s*)|(\s*([-|\(|\*|\~]?)\s*((now\s*open)|(opening)|(reopening))\s*(.*)$)", " ", name)

        # remove ["location", "new location","two locations", "relocation"]
        name = re.sub("(\s*([-|\(|\*|\~]?)\s*((new\s*location(s?))|(location(s?))|(two location(s?)|(relocation(s?))\s*(.*)$))\s*(.*)$)", "", name)

        # remove \s*-\s*  and all after words
        name = re.sub("(\s+\-\s+.*)$", "", name)

        # replace - to space
        name = re.sub("\-", " ", name)

        # remove  lower level  upper level  level 2  2nd level
        name = re.sub("(^((level)|(next level))\s*(\d?)\s+)", "", name)
        name = re.sub("(\s*((level)|(upper level)|(lower level))\s*(\d?)(.*)$)", "", name)

        #change common stores name to the same
        common_stores = ["aldo" , "starbucks" , "att", "aaa", "advance america", "as seen on tv", "sanrio", "hollister", "five guy", "rubios", "ecoatm", "hooter", "joppa", "wasabi", "guitar center"," rainforest cafe", "relax the back", "uno chicago grill","nys collection"]
        for common_store in common_stores:
            if(re.search(common_store, name)):
                name = common_store

        # remove other special characters
        # *, #, !, ?, ', @,  $, +, ;
        name = re.sub("(\s*)[\.|\,|\\\"|\\\'|\(|\)|\?|\@|\$|\+|\;|\\'|\\\"|\!|\*|\#](\s*)", " ", name)

        if not name in stores_dic.keys():
            #calculate the word frequency here, store them and try manually analyse the repeat name
            name_tokens = name.split(" ")
            for i in name_tokens:
                if i in store_name_frequency.keys():
                    store_name_frequency[i] += 1
                else:
                    store_name_frequency[i] = 1
            stores_dic[name] = 1
store_file.close()

with open(final_file, 'wb') as file:
    writer = csv.writer(file, delimiter=',')
    for key in stores_dic.keys():
        writer.writerow([key])
file.close()

print store_name_frequency
with open(dir + "frequency.csv", "wb") as file:
    writer = csv.writer(file, delimiter=',')
    for key, val in store_name_frequency.items():
        writer.writerow([key, val])
file.close()

# write frequency file to manually remove some data




