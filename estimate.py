
import os
import os.path
import numpy as np
import utils
from sklearn import ensemble, cross_validation, linear_model
from sklearn.utils import shuffle
import numpy.core.defchararray as ncd
from joblib import Parallel, delayed, cpu_count
import time
import subprocess
import sys
import pandas as pd

_samples = 800
# _n_clean = 190
_hash_id = os.urandom(6).encode('hex')

def set_ranges(probs):
    probs[np.bitwise_and(probs > 0.0, probs < 0.1)] = 0.1
    probs[np.bitwise_and(probs > 0.1, probs < 0.2)] = 0.2
    probs[np.bitwise_and(probs > 0.2, probs < 0.3)] = 0.3
    probs[np.bitwise_and(probs > 0.3, probs < 0.4)] = 0.4
    probs[np.bitwise_and(probs > 0.4, probs < 0.5)] = 0.5
    probs[np.bitwise_and(probs > 0.5, probs < 0.6)] = 0.6
    probs[np.bitwise_and(probs > 0.6, probs < 0.7)] = 0.7
    probs[np.bitwise_and(probs > 0.7, probs < 0.8)] = 0.8
    probs[np.bitwise_and(probs > 0.8, probs < 0.9)] = 0.9
    probs[np.bitwise_and(probs > 0.9, probs < 1.0)] = 1.0


def p_ill_do_it_faggot(d, X_out, y_out):
    Xt, yt = utils.load_driver_pca(d)
    X_out, y_out = utils.load_outliers_pca(d, _samples-200)
    yt[:] = 1
    # rs = utils.clean_noise(Xt, _n_clean)
    # Xt = np.vstack((rs["X_clean"], rs["X_noise"]))
    # yt[_n_clean:] = 0
    X = np.vstack((Xt, X_out))
    y = np.hstack((yt, y_out))
    k_fold = cross_validation.KFold(len(X), n_folds=5)
    X, y = shuffle(X, y, random_state=13)
    d_p = np.zeros((200))
    for j, (train, test) in enumerate(k_fold):
        # probas_ = gbrt.fit(X[train], y[train]).predict_proba(X[test])
        gbrt.fit(X[train], y[train])
        regr.fit(X[train], y[train])
        my_p = gbrt.predict_proba(Xt)[:, 1] + regr.predict_proba(Xt)[:, 1]
        d_p += my_p
    d_p /= float(len(k_fold) * 2)

    d_p = (d_p - d_p.min())/(d_p.max()-d_p.min())

    d_p[d_p > 0.9] = 1.0
    d_p[d_p < 0.1] = 0.0

    indexes = np.arange(1, 201).astype(str)
    d_id = (np.ones(200)*int(d)).astype(int).astype(str)
    und = np.ones(200).astype(str)
    und[:] = "_"
    first_column = ncd.add(d_id, und)
    first_column = ncd.add(first_column, indexes)

    second_column = np.array(["%.8f" % p for p in d_p])

    outp = np.vstack((first_column, second_column)).T

    np.savetxt("subm_%s/%s.csv" % (_hash_id, d), outp, fmt="%s", delimiter=",")
    print d


if __name__ == "__main__":
    curr_fold = ""
    done = []
    if len(sys.argv) > 1:
        curr_fold = sys.argv[1]
        _hash_id = curr_fold[5:]
        done = os.listdir("%s" % curr_fold)
        done = filter(lambda idd: idd[0] != ".", done)  # remove non files
        done = map(lambda x: x[:-4], done)  # remove csv extension

    folders = os.listdir("data/drivers/")
    drivers = filter(lambda idd: idd[0] != "." and int(idd), folders)
    to_do = sorted(list(set(drivers) - set(done)))

    params = {
        'n_estimators': 400,
        'max_depth': 3,
        'learning_rate': 0.05
    }
    params2 = {
        'C': 1.0,
        'penalty': 'l1'
    }

    regr = linear_model.LogisticRegression(**params2)
    gbrt = ensemble.GradientBoostingClassifier(**params)
    Xf, yf = utils.load_outliers_pca("1", _samples-200)

    if not os.path.exists("subm_%s" % _hash_id):
        os.makedirs("subm_%s" % _hash_id)
    print "estimating %s drivers on %s cpus" % (len(to_do), cpu_count())
    n_proc = cpu_count()
    start = time.time()
    Parallel(n_jobs=n_proc)(delayed(p_ill_do_it_faggot)(d, Xf, yf) for (ind, d) in enumerate(to_do))

    repeat_flag = False
    if len(sys.argv) > 2:
        repeat_flag = (sys.argv[2] == "repeat")

    if repeat_flag:
        subprocess.call("python fix_repeated.py subm_%s" % _hash_id, shell=True)

    subprocess.call("python create_subm.py subm_%s" % _hash_id, shell=True)

    print str(time.time()-start)
    print "estimated and saved"

