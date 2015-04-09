__author__ = 'lily'

import sys

from pyspark.context import SparkContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD

def logistic_regression(store, trianing, testing):
     # analyze the result for one store in malls
    #  https://github.com/apache/spark/blob/master/examples/src/main/python/mllib/logistic_regression.py
    model = LogisticRegressionWithSGD.train(points, iterations)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print >> sys.stderr, "Usage: logistic_regression"
        exit(1)
    sc = SparkContext(appName="Logistic regression")

    # Load and parse the data file into an RDD of LabeledPoint.
    # mall data
    dir = ""
    file = dir + "malls.csv"
    malls_data = MLUtils.loadLibSVMFile(sc, file)
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = malls_data.randomSplit([0.7, 0.3])

    # load store data here
    stores = file
    for i in stores:
        print('\nRunning logistic regression \n')
        logistic_regression(i, trainingData, testData)
    sc.stop()