import numpy as np
import bz2
import textdistance


# NCD from NohGenerator
def compute_ncd(a, b):
    ca = float(len(bz2.compress(a.encode())))
    cb = float(len(bz2.compress(b.encode())))
    cab = float(len(bz2.compress((a + b).encode())))
    return (cab - min(ca, cb)) / max(ca, cb)


# for genotype (weights arr. - weights arr.)
# similar = 1, dissimilar = 0
def norm_similarity(ind1, ind2):
    # using euclidean distance
    a = np.array(ind1)
    b = np.array(ind2)
    # NB: distance between two sum-to-1 array is between 0 and (2**0.5)
    # rearranged between [0,1] -> (val - min)/ (max-min) => val/max with min=0 and max = (2**0.5)
    return ((2**0.5)-np.sum((a - b) ** 2) ** 0.5)/(2**0.5)


# similar = 0, dissimilar = 1
def norm_dissimilarity(ind1, ind2):
    return 1 - norm_similarity(ind1, ind2)


# for phenotype (string - string)
# similar = 1, dissimilar = 0
def str_similarity(a, b):
    # jacc = textdistance.jaccard(a.split(" "), b.split(" "))
    # ham = textdistance.hamming.normalized_similarity(a,b)
    # lev = textdistance.levenshtein.normalized_similarity(a,b)
    # jw1 = textdistance.jaro_winkler(a,b)
    # ro = textdistance.ratcliff_obershelp(a,b)
    # return (textdistance.jaro_winkler(a,b) + 1 - distance.jaccard(a.split(" "), b.split(" ")  )) / 2
    return textdistance.jaccard(a, b)


# similar = 0, dissimilar = 1
def str_dissimilarity(a, b):
    return 1 - str_similarity(a,b)
