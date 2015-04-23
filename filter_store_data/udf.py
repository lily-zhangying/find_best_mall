import unicodedata
import re


@schemaFunction("remove_accent_marks")
def remove_accent_marks(input_str):
        nkfd_form = unicodedata.normalize('NFKD', unicode(input_str))
        return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])

@outputSchema("t:tuple(name,mall_id)")
def filter(name,mall_id):
	mall_id = unicode(mall_id, "utf8").lower().strip()
        if(mall_id == "none"):
		return (name, -1)
	name = unicode(name, "utf8").lower().strip()
	# replace several spaces to one space
        name = re.sub("\s+", " ", name)
        # remove special characters
        name = re.sub("(\s*)[&|'|\\\|\/](\s*)", "", name)
        #replace accent_marks of name
        name = remove_accent_marks(name)
        #filter atm & vending machines
        if (re.search("(\s*atm\s*)|(\s*vending\s*machines\s*)|(^(advanced)\s*)", name)):
            return (name, -1)
        #filter open EXCEPT ["open advanced", "mri open mobile"]
        if (re.search("(^(open)\s*((?!(mobile))|(?!(advanced mri)))$)", name)):
            return (name, -1)

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
        name = re.sub("(\s*)[\.|\,|\\\%|\\\"|\\\'|\(|\)|\?|\@|\$|\+|\;|\\'|\\\"|\{|\}|\!|\*|\#](\s*)", " ", name)

        if(len(re.sub("\s*", "", name)) <= 0):
            return (name, -1)

        if(len(re.sub("\s*", "", name)) > 0):
            return (name, mall_id)
