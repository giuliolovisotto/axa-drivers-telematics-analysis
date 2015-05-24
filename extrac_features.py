__author__ = 'giulio'

import sys
import os
import utils
import time
import subprocess

if __name__ == "__main__":
    if len(sys.argv) > 1:
        first, last, pca = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    else:
        first, last = 0, 4000
    start = time.time()

    folders = os.listdir("data/drivers/")
    drivers = filter(lambda idd: idd[0] != "." and first <= int(idd) <= last, folders)
    print "processing %s drivers" % len(drivers)

    samples = utils.extract_features_drivers(drivers)

    print str(time.time()-start)

    del samples

    if pca == "pca":
        subprocess.call("python pca.py", shell=True)

    print "processed and saved"

