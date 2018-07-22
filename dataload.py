import random

from pyspark.ml import Pipeline
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkContext
import numpy as np
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer


newsgroups_train = fetch_20newsgroups(subset='test')

sc = SparkContext()

sqlContext = SQLContext(sc)
data=dict(zip(newsgroups_train["data"],newsgroups_train["target"]))
print len(data)
dataframe=pd.DataFrame(data.items(), columns=['Text', 'Target'])
data=sqlContext.createDataFrame(dataframe)
data.printSchema()
data = data.na.drop(thresh=2)

regexTokenizer = RegexTokenizer(inputCol="Text", outputCol="words", pattern="\\W")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = CountVectorizer(inputCol="filtered", outputCol="rawFeatures")
out_data= IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover,hashingTF,out_data])
pipelineFit = pipeline.fit(data)
pipelineFit=pipelineFit.transform(data)

"""
RDD:
id: sparsevec
"""
pipelineFit.persist()

processed_data = pipelineFit.select("features").rdd
#.map(lambda x:[x[0].toArray()]).toDF(["features"]).rdd
rows_20 = processed_data.take(20)
means = map(lambda x:x.features,rows_20)
def get_closest(v1,means):
    smallest = 999999999999
    for index,mean in enumerate(means):
        dist = v1.squared_distance(mean)
        if dist<smallest:
            smallest = dist
            smallest_ind = index
    return smallest_ind
def assign_labels(means):
    print "222222222222222"
    ret_value = processed_data.map(lambda x: (get_closest(x.features,means),) + x).toDF(["label","features"]).rdd
    print "3333333333333"
    return ret_value


def compute_means(rdd):
    print "5555555555555555"
    new_sums = rdd.mapValues(lambda v: v.toArray()).reduceByKey(lambda x, y: x + y)
    new_sums.persist()
    sums_ = new_sums.collectAsMap()
    counts = rdd.countByKey()
    means = {i: sums_[i] / counts[i] for i in sums_}
    new_means = [means[i] for i in sorted(means)]
    print "xxxxxxxxxxxxxx"
    return new_means

i=1
while i<=20:
    print "111111111111"
    labeled=assign_labels(means)
    print "44444444444444"
    means = compute_means(labeled)
    print "666666666666666"
    i = i + 1
    print(i)