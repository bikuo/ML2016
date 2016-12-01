from __future__ import print_function
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
import string
import logging
from optparse import OptionParser
import sys
from time import time
import numpy as np
import math
from scipy.spatial.distance import cosine
import os
import re

def load_titles():
    doc = []
    with open('title_StackOverflow.txt','r') as f:
        doc = f.readlines()
    for i in range(len(doc)):
        doc[i] = doc[i].rstrip('\n').lower()
        doc[i] = re.sub('[^a-z ]+','',doc[i])
        # do some lemmatizing
        if 'ajaxifying' in doc[i]:
            tmp = doc[i].split('ajaxifying')
            doc[i] = tmp[0]+'ajax'+tmp[1]
        if 'ajaxify' in doc[i]:
            tmp = doc[i].split('ajaxify')
            doc[i] = tmp[0]+'ajax'+tmp[1]
        if 'linqify' in doc[i]:
            tmp = doc[i].split('linqify')
            doc[i] = tmp[0]+'linq'+tmp[1]
    return doc

os.chdir(sys.argv[1])
output = sys.argv[2]
###############################################################################
dataset = load_titles()

#TF-IDF
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
X = vectorizer.fit_transform(dataset)

#SVD & LSA
lsa_components = 20
svd = TruncatedSVD(lsa_components)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)

###############################################################################
# Do the actual Kmeans clustering
clusters = 19
km = KMeans(n_clusters=clusters, init='k-means++', max_iter=100, n_init=100, verbose=False)
km.fit(X)
Y = km.predict(X)
###
# generate tags for each cluster
#
print("Top terms per cluster:")
original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
freq = np.array(original_space_centroids)
freq.sort()
freq = freq[:,::-1]
terms = vectorizer.get_feature_names()
good_tags = []
for i in range(clusters):
    cluster_terms = []
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % (terms[ind]), end='')
    cluster_terms.append(terms[order_centroids[i,0]].encode('utf-8').lower())
    j=1
    while(freq[i,j] > 0.9*freq[i,j-1]):
        cluster_terms.append(terms[order_centroids[i,j]].encode('utf-8').lower())
        j+=1
    good_tags.append(cluster_terms)
    print ('\n')
print (good_tags)

result = np.empty((5000000,2),dtype = 'int')
test = pd.read_csv('check_index.csv')
ind_pair = test.as_matrix(columns = test.columns[1:])

#prediction
for i in range(ind_pair.shape[0]):
    if i % 50000 == 0:
        print (i)
    same = 0
    x = ind_pair[i][0]
    y = ind_pair[i][1]
    for tags in good_tags:
        a = all([tag in dataset[x] for tag in tags])
        b = all([tag in dataset[y] for tag in tags])
        if a and b and cosine(X[x],X[y]) < 0.33 and Y[x] == Y[y]:
            same = 1    
            break
    result[i] = np.array([i,same])
np.savetxt(output, result, delimiter = ',', fmt = '%d', header = 'ID,Ans', comments = '')   
print(sum(result[1:,1]))
if __name__ == '__main__':
    pass