from __future__ import print_function

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from optparse import OptionParser
import sys
from time import time
import numpy as np

# parse commandline arguments
op = OptionParser()
op.add_option("-f", "--file",
              dest="filename", 
              help="input filename")
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

(opts, args) = op.parse_args()

if len(args) > 0:
    op.error("this script takes no arguments.")
    print(__doc__)
    op.print_help()
    sys.exit(1)

sections = [];
with open(opts.filename) as f:
    sections = f.readlines()

nrLines = len(sections)

vectorizer = TfidfVectorizer(min_df=3,max_df=0.5,stop_words='english')
X = vectorizer.fit_transform(sections)
print("n_samples: %d, n_features: %d" % X.shape)
true_k = 9

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
# print("Clustering sparse data with %s" % model)

model.fit(X)
XPredLables = model.labels_

for i in range(200):
    print (XPredLables[i], "\t", sections[i])


if opts.n_components:
    original_space_centroids = svd.inverse_transform(model.cluster_centers_)
    space_centroids = original_space_centroids
    order_centroids = original_space_centroids.argsort()[:, ::-1]
else:
    space_centroids = model.cluster_centers_
    order_centroids = space_centroids.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
print("Top terms per cluster:")
for i in range(true_k):
    print ("Cluster %d:" % i)
    for ind in order_centroids[i, :20]:
        print (format(terms[ind]), "\t", space_centroids[i, ind])
    print()


def main():

if __name__ == '__main__': main()    

