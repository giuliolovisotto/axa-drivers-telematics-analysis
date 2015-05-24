__author__ = 'giulio'


import numpy as np

from joblib import Parallel, delayed, cpu_count
from sklearn import decomposition, preprocessing
import os
import utils
import time
import sys

# number of components of the feature space after pca is in utils

from tempfile import mkstemp
from shutil import move
from os import remove, close


def p_save_reduced(d, x):
    np.savetxt("data/reduced/%s.csv" % d, x)
    print d
    

def save_pcaed_feat(drivers_list, X):
    if len(drivers_list) == 0:
        raise Exception("empty")
    
    n_proc = cpu_count()

    Parallel(n_jobs=n_proc)(delayed(p_save_reduced)(d, X[ind*200:(ind+1)*200, :]) for (ind, d) in enumerate(drivers_list))
    return True

if __name__ == "__main__":
    start = time.time()
    folders = os.listdir("data/drivers/")
    drivers = filter(lambda idd: idd[0] != "." and 0<int(idd)<4000, folders)
    print "reducing %s drivers" % len(drivers)
    X, _ = utils.load_data()
    X = preprocessing.scale(X)
    if len(sys.argv) > 1:
        cols = int(sys.argv[1])
        X = np.delete(X, cols, 1)
    pca = decomposition.PCA()
    pca.fit(X)
    pca.n_components = pca.explained_variance_[pca.explained_variance_ > 1e-05].shape[0]

    X = pca.fit_transform(X)
    save_pcaed_feat(drivers, X)
    print str(time.time()-start)

    # rewrite _COMP_PCA in utils
    # the line to look for is
    old = "_COMP_PCA = %s" % str(utils._COMP_PCA)
    new = "_COMP_PCA = %s" % str(pca.n_components)
    utils.replace("utils.py", old, new)
    print "reduced to %s-dim and saved" % str(pca.n_components)
    
