# filter the store data
# remove the redundant store's name
import csv
import re
import unicodedata
import sys
reload(sys)
import operator
sys.setdefaultencoding("utf-8")

root = "/Users/lily/workspace/find_best_mall/filter_store_data"
dir = root + "/dataset"
file = dir + "/store.csv"
final_file = dir + "/sort_final_store.csv"
store_id_file = dir + "/sort_store_id_file.csv"
stores_list = []
stores_dic = {}

def remove_accent_marks(input_str):
    nkfd_form = unicodedata.normalize('NFKD', unicode(input_str))
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])

def get_rules(name):
    file = root + "/filter_rules/" + name + "_name_rules.csv"
    tmp = []
    with open(file, "rU") as f:
        reader = csv.reader(f)
        for row in reader:
            tmp.append(row[0])
    f.close()
    return tmp

 # delete role
delete_rules = get_rules("delete")

# sub rule
sub_rules = get_rules("sub")
# change rule
change_rules = get_rules("change")
# print change_rules

with open(file, 'rU') as store_file:
    reader = csv.reader(store_file, delimiter=",")
    for row in reader:
        # lowercase and trim string
        mall_id = row[2].lower().strip()
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
        common_stores = ["aldo" , "starbucks" , "att", "aaa", "advance america", "as seen on tv", "sanrio", "hollister", "five guy", "rubios", "ecoatm", "hooter", "joppa", "wasabi", "guitar center"," rainforest cafe", "relax the back", "uno chicago grill","nys collection", "tmobile", "macdonald", "verizon"]
        for common_store in common_stores:
            if(re.search(common_store, name)):
                name = common_store

        # remove other special characters
        # *, #, !, ?, ', @,  $, +, ; % { }
        name = re.sub("(\s*)[\.|\,|\\\%|\\\"|\\\'|\(|\)|\?|\@|\$|\+|\;|\\'|\\\"|\{|\}|\!|\*|\#|\:|\;|\\'|\!|\\\](\s*)", " ", name)

        name = re.sub("\s+", " ", name)
        name = name.strip()

        # delete rule
        # meet these words in delte rules, just ignore them
        IS_IGNORE = False
        for rule in delete_rules:
            if (re.search(rule, name) != None):
                IS_IGNORE = True
                break
        if(IS_IGNORE):
            continue

        # sub rule
        # meet sub rules, replace the origin names with the sub sule name
        for rule in sub_rules:
            if (re.search(rule, name) != None):
                new_rule = re.sub("\^", "", rule)
                name = new_rule
                break

        # change rule
        # change store name from A -> B
        change_rules = get_rules("change")
        for rule in change_rules:
            rule = rule.split("->")
            rule_from = rule[0].strip()
            # rule_from = pattern = re.compile(re.escape(rule[0].strip()))
            rule_to = rule[1].strip()
            name = re.sub(rule_from, rule_to, name)

        # if the name just have spaces, ignore
        if(len(re.sub("\s*", "", name)) <= 0):
            continue

        if(len(re.sub("\s*", "", name)) > 0):
            # create unique id for stores
            if not(name in stores_dic):
                store_id = len(stores_dic)
                stores_dic[name] = store_id
            stores_list.append([name, stores_dic[name],mall_id])
store_file.close()


with open(final_file, 'wb') as file:
    writer = csv.writer(file, delimiter=',')
    my_list = sorted(stores_list, key=operator.itemgetter(1))
    writer.writerow(["store_name", "store_new_id", "mall_id"])
    for val in my_list:
        writer.writerow(val)
file.close()

with open(store_id_file, "wb") as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["store_name", "store_new_id"])
    my_dic = sorted(stores_dic.items(), key=lambda x: x[1])
    for val in my_dic:
        writer.writerow(val)
file.close()





