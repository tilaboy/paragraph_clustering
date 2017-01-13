from __future__ import print_function

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import argparse
import sys
from time import time
import numpy as np
import csv

def read_documents(filename):
    sections = [];
    tags = [];
    with open(filename) as f:
        linereader = csv.reader(f, delimiter="\t")
        for row in linereader:
            sections.append(row[1])
            tags.append(row[0])
    return (sections, tags)

def train(sections, opts):

    # min_df, max_df can be in the range of 0~1, or integers 
    vectorizer = TfidfVectorizer(min_df=3,max_df=0.5,stop_words='english')
    X = vectorizer.fit_transform(sections)
    print("n_samples: %d, n_features: %d" % X.shape)
    true_k = opts.true_k

    if opts.n_components:
        print("Performing dimensionality reduction using LSA")
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

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(model.cluster_centers_)
        space_centroids = original_space_centroids
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        space_centroids = model.cluster_centers_
        order_centroids = space_centroids.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    print("Top terms per cluster:")
    true_k_list = ['{:10s}'.format(str(i)) for i in range(true_k)]
    print("{:8s}\t{}".format("cluster","\t".join(true_k_list)))
    for i in range(10):
        top_terms_per_cluster = [ '{:10s}'.format(terms[ind]) for ind in order_centroids[:, i] ]
        print ("{:8s}\t{}".format(str(i), "\t".join(top_terms_per_cluster)))
    print()

    # for i in range(true_k):
    #     print ("Cluster %d:" % i)
    #     for ind in order_centroids[i, :20]:
    #         print (format(terms[ind]), "\t", space_centroids[i, ind])
    #     print()

    return (vectorizer, model)


def check_accuracy (predTags, trueTags, true_k):
    nrLines = len(predTags)
    assert nrLines == len(trueTags)
    uniqTags = list(set(trueTags))
    summary = { tag:{} for tag in uniqTags }
    for i in range(nrLines):
        tag = trueTags[i]
        predTag = predTags[i]
        summary[tag][predTag] = summary[tag].get(predTag, 0) + 1
    print ("{:20s}\t{}\t{}".format("tag","true","\t".join(map(str,list(range(true_k))))))
    for tag in uniqTags:
        predTags = []
        for predTag in range(true_k):
            nr_predTag = summary[tag].get(predTag, 0)
            predTags.append(nr_predTag)
        print("{:20s}\t{}\t{}".format(tag,sum(summary[tag].values()),"\t".join(map(str,predTags))))
    print()
    
    # print (XPredLables[i], "\t", sections[i])

def check_samples (predTags, trueTags, sections, tagToCheck=['personalsec','educationsec','experiencesec','skillsec']):
    nrLines = len(predTags)
    assert nrLines == len(trueTags)
    uniqTags = list(set(trueTags))
    
    print ("First 200 sections")
    for i in range(200):
        print ("{:20s}{:4s}{}".format(trueTags[i], str(predTags[i]),sections[i]))
    print()

    # for each section, output the 
    



def main():
    # parse commandline arguments
    op = argparse.ArgumentParser(description='read input file, train a clustering model, and test on test data')
    op.add_argument("-a" ,"--action", dest="action", help="support action 'train', 'test'")
    op.add_argument("-f", "--file",
                  dest="filename", 
                  help="input filename")
    op.add_argument("--class",
                  dest="true_k", type=int, default=9,
                  help="number of clussters")
    op.add_argument("--lsa",
                  dest="n_components", type=int, 
                  help="Preprocess documents with latent semantic analysis.")
    op.add_argument("-v", "--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Print progress reports while training/testing.")

    ops = op.parse_args()

    if ( ops.action == 'train' ):
        print ("start training clustering model")
        (train_sections, train_tags) = read_documents(ops.filename)
        (verctorizer, model) = train(train_sections, ops)
        check_accuracy(model.labels_, train_tags, ops.true_k)
        check_samples(model.labels_, train_tags, train_sections)
        # 
    elif ( ops.action == 'test' ):
        print ("start testing clustering model")
        sections = read_documents(ops.filename)
        test()
    elif ( ops.action ):
        print ("ERROR: unknown action")
        op.print_help()
    else:
        print ("ERROR: Missing action")
        op.print_help()


if __name__ == '__main__': main()    

