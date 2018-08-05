import random
from pyspark.ml import Pipeline
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkContext
import numpy as np
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer
from sklearn.decomposition import PCA
import csv
newsgroups_train = fetch_20newsgroups(subset='train')
sc = SparkContext("local[4]", "Kmeans Clustering")
sqlContext = SQLContext(sc)
data=dict(zip(newsgroups_train["data"],newsgroups_train["target"]))
print len(data)
dataframe=pd.DataFrame(data.items(), columns=['Text_NotTransformed', 'Target'])
sentence=dataframe["Text_NotTransformed"]
print sentence
sentence = sentence.str.lower()
sentence = sentence.str.replace(r"[^a-zA-Z0-9]+"," ")
sentence = sentence.str.replace(r"([^\w])"," ")
sentence = sentence.str.replace(r"\b\d+\b", " ")
sentence = sentence.str.replace(r"\s+|\r|\n", " ")
sentence = sentence.str.replace(r"^\s+|\s$", "")
se = pd.Series(sentence)
dataframe["Text"]=se.values
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
rows_6 = processed_data.takeSample(False,6, seed = 12L)
means = map(lambda x:x.features,rows_6)
print (means)
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

j=0
print j
def compute_means(rdd):
    print "5555555555555555"
    new_sums = rdd.mapValues(lambda v: [v.toArray(), 1]).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    print "don't know don't know"
    newPoints = new_sums.map(
        lambda st: (st[0], st[1][0] / st[1][1])).collectAsMap()

    print "over over"
    return newPoints
while j<40:
    print "111111111111"
    labeled=assign_labels(means)
    print "44444444444444"
    means = compute_means(labeled)
    means = [means[i] for i in sorted(means)]
    print "666666666666666"
    j = j + 1
    print j


cc=assign_labels(means).collect()
traget=[]
data=[]
for i in range(6):
    z=0
    for k in range(len(cc)):
        if(z>10 ):
            break
        if(cc[k][0]==i):
            data.append(cc[k][1])
            traget.append(cc[k][0])
            z=z+1

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(data)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
pca1=[]
pca2=[]
pca3=[]
for i in range(len(principalComponents)):
    pca1.append(principalComponents[i][0])
    pca2.append(principalComponents[i][1])
    pca3.append(principalComponents[i][2])

pca1=np.array(pca1)
pca2=np.array(pca2)
pca3=np.array(pca3)


label = np.array(traget)
df=pd.DataFrame({"pca1":pca1,"pca2":pca2,"pca3":pca3,"label":label})
df.to_csv("submission.csv", index = False)

colors = {0 : 'b',
          1 : 'y',
          2 : 'r',
          3 : 'c',
          4 : 'g',
          5 : 'm'}



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

c = [colors[val] for val in label]
ax.scatter(pca1, pca2, pca3, c=c,marker='o')
plt.show()

