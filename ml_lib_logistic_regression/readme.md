## description

* use logistic regression to analyze

## tech
* use spark ml lib (https://spark.apache.org/docs/1.1.0/mllib-guide.html)
* logistic regression(https://spark.apache.org/docs/1.1.0/mllib-linear-methods.html#logistic-regression)

## run on local machine

* install spark first(mac)
```bash
brew install apache-spark
```
* run python file
```bash
$ spark-submit --master local logistic.py
```

## run on cluster
```bash
spark-submit help
```

## in case I cannot find
http://spark.apache.org/docs/1.2.1/quick-start.html#self-contained-applications
