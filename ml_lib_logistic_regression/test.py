__author__ = 'lily'


"""SimpleApp.py"""
from pyspark import SparkContext

logFile = "/Users/lily/workspace/find_best_mall/logistic regression/readme.md"  # Should be some file on your system
sc = SparkContext("local", "Simple App")
logData = sc.textFile(logFile).cache()

numAs = logData.filter(lambda s: 'a' in s).count()
numBs = logData.filter(lambda s: 'b' in s).count()

print "Lines with a: %i, lines with b: %i" % (numAs, numBs)