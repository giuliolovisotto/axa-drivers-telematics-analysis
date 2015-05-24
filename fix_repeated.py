__author__ = 'giulio'

import os
import sys
from joblib import Parallel, delayed, cpu_count
import numpy as np
import numpy.core.defchararray as ncd
import pandas as pd

folders = os.listdir("data/drivers/")

drivers = filter(lambda idd: idd[0] != "." and 0 < int(idd), folders)

direct = sys.argv[1]

def paral(path, d):
    probs = pd.read_csv("%s/%s.csv" % (direct, d), header=None).as_matrix()[:, 1].astype(float)
    # probs = np.loadtxt("%s/%s.csv" % (direct, d))[:, 1].astype(float)
    repeat = np.loadtxt("rep_trips/%s.txt" % d)[:, 1].astype(int)
    new_p = probs
    new_p[np.bitwise_and(repeat == 1, 1 > 0.5)] = 1.0
    indexes = np.arange(1, 201).astype(str)
    d_id = (np.ones(200)*int(d)).astype(int).astype(str)
    und = np.ones(200).astype(str)
    und[:] = "_"
    first_column = ncd.add(d_id, und)
    first_column = ncd.add(first_column, indexes)
    second_column = np.array(["%.8f" % p for p in new_p])

    outp = np.vstack((first_column, second_column)).T
    np.savetxt("%s/%s.csv" % (path, d), outp, fmt="%s", delimiter=",")
    print d


Parallel(n_jobs=cpu_count())(delayed(paral)(direct, d) for d in drivers)



