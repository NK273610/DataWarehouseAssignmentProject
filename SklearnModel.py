from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import os
from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset='train')
data=dict(zip(newsgroups_train["data"],newsgroups_train["target"]))
dataframe=pd.DataFrame(data.items(), columns=['Text', 'Target'])
sentence=dataframe["Text"]

sentence = sentence.str.lower()
sentence = sentence.str.replace(r"[^a-zA-Z0-9]+"," ")
sentence = sentence.str.replace(r"([^\w])"," ")
sentence = sentence.str.replace(r"\b\d+\b", " ")
sentence = sentence.str.replace(r"\s+|\r|\n", " ")
sentence = sentence.str.replace(r"^\s+|\s$", "")

from nltk.tokenize import RegexpTokenizer
STEMMER = PorterStemmer()
STOP_WORD = stopwords.words('english')
TOKENIZER = RegexpTokenizer(r'\w+')
def textprocessing(text):
    return ' '.join(STEMMER.stem(token)  for token in TOKENIZER.tokenize(text.lower()) if token not in STOP_WORD and len(token) > 1)

final=[]
i=0
for data in sentence:

    print i
    final.append(textprocessing(data))
    i=i+1

se = pd.Series(final)
print se
dataframe["processed_text"]=se.values
tfidf_vectorizer = TfidfVectorizer(max_features=1000000, strip_accents='unicode', analyzer='word',lowercase=True, use_idf=True)

# Fit the vectorizer to text data
tfidf_matrix = tfidf_vectorizer.fit_transform(dataframe["processed_text"])


# Kmeans++
km = KMeans(n_clusters=20, init='k-means++', max_iter=300, n_init=1, verbose=0, random_state=3425)
km.fit(tfidf_matrix)
labels = km.labels_
clusters = labels.tolist()
print clusters
tfidf_matrix=tfidf_matrix.toarray()
print tfidf_matrix
traget=[]
data2=[]
for i in range(6):
    z=0
    for k in range(tfidf_matrix.shape[0]):
        if(z>10 ):
            break
        if(clusters[k]==i):
            data2.append(tfidf_matrix[k])
            traget.append(clusters[k])
            z=z+1

print len(data2)

pca = PCA(n_components=3)

principalComponents = pca.fit_transform(data2)

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
df.to_csv("submission2.csv", index = False)

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





