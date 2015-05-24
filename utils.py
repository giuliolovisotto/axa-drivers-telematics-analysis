__author__ = 'giulio'

import pandas as pd
import numpy as np
import math
from joblib import Parallel, delayed, cpu_count
import sklearn
import sklearn.cluster
import os
import random
from tempfile import mkstemp
from shutil import move
from os import remove, close


# TODO: separa brake e acceleration e metti threshold per considerarli
# TODO: tempo fermi

_N_F = 62
_COMP_PCA = 57

_SPEED_MAX = 40.0
_SPEED_MIN = 0.0
_SPEED_BINS = np.linspace(_SPEED_MIN, _SPEED_MAX, 11)

_ACCEL_MIN = -35.0
_ACCEL_MAX = 35.0
_ACCEL_BINS = np.linspace(_ACCEL_MIN, _ACCEL_MAX, 15)

# _BRAKING_BINS = np.linspace(_ACCEL_MIN, 0.0, 6)

_JERK_MIN = -2.0
_JERK_MAX = 2.0
_JERK_BINS = np.linspace(_JERK_MIN, _JERK_MAX, 13)

_SNAP_MIN = -3.5
_SNAP_MAX = 3.5
_SNAP_BINS = np.linspace(_SNAP_MIN, _SNAP_MAX, 9)

_CACCEL_MIN = 0.0
_CACCEL_MAX = np.pi*_SPEED_MAX
_CACCEL_BINS = np.linspace(_CACCEL_MIN, _CACCEL_MAX, 11)

_ANGLE_MIN = -np.pi
_ANGLE_MAX = np.pi
_ANGLE_BINS = np.linspace(_ANGLE_MIN, _ANGLE_MAX, 11)

_A_SPEED_MIN = -2 * np.pi
_A_SPEED_MAX = 2 * np.pi
_A_SPEED_BINS = np.linspace(_A_SPEED_MIN, _A_SPEED_MAX, 11)


def replace(file_path, pattern, subst):
    # Create temp file
    fh, abs_path = mkstemp()
    new_file = open(abs_path, 'w')
    old_file = open(file_path)
    for line in old_file:
        new_file.write(line.replace(pattern, subst))
    # close temp file
    new_file.close()
    close(fh)
    old_file.close()
    # Remove original file
    remove(file_path)
    # Move new file
    move(abs_path, file_path)

def extract_features_drivers(drivers_list):
    if len(drivers_list) == 0:
        raise Exception("empty")

    features = np.memmap("output", dtype="float64", shape=(len(drivers_list), 200, _N_F), mode='w+')

    n_proc = cpu_count()
    Parallel(n_jobs=n_proc)(delayed(p_getinfo)(d, features, i) for i, d in enumerate(drivers_list))
    return features


def p_getinfo(d, feat, i):
    r = get_info_for_driver(d)
    feat[i] = r
    np.savetxt("data/features/%s.csv" % d, r)
    print d


def get_info_for_trip(tripfile):
    """
    ok we want
    1) tangential speed (avg, min, max, var)
    2) tangential acceleration (avg, min, max, var)
    3) angular speed (avg, min, max, var)
    4) angular acceleration (avg, min, max, var)
    5) centripetal acceleration (avg, min, max, var)
    6) time
    7) distance
    :param tripfile:
    :return:
    """
    trip = pd.read_csv(tripfile)

    t_s = np.zeros([len(trip)])
    t_a = np.zeros([len(trip)])
    t_j = np.zeros([len(trip)])
    t_snap = np.zeros([len(trip)])
    # t_d_v_s = np.zeros([len(trip)])

    # we store the signed angle with respect to x axis v = [1, 0] at every step to get angular speed
    '''
    angles = np.zeros([len(trip)])

    a_s = np.zeros([len(trip)])
    a_a = np.zeros([len(trip)])
    c_a = np.zeros([len(trip)])
    '''
    time = 0.0
    distance = 0.0

    cur_pos = trip[:1].as_matrix().flatten()
    cur_dir = np.zeros(2)
    prev_pos = cur_pos
    prev_dir = np.zeros(2)

    for i, pos in trip[1:].iterrows():
        cur_pos = pos.as_matrix().flatten()

        t_s[i] = math.sqrt((cur_pos[0]-prev_pos[0])**2 + (cur_pos[1]-prev_pos[1])**2)

        if _SPEED_MIN > t_s[i]:
            t_s[i] = _SPEED_MIN
        if _SPEED_MAX < t_s[i]:
            t_s[i] = _SPEED_MAX

        t_a[i] = t_s[i] - t_s[i-1]
        if _ACCEL_MIN > t_a[i]:
            t_a[i] = _ACCEL_MIN
        if _ACCEL_MAX < t_a[i]:
            t_a[i] = _ACCEL_MAX
        '''
        t_d_v_s[i] = (t_a[i]/t_s[i]) if t_s[i] != 0.0 else 0.0
        if t_d_v_s[i] < -1.0:
            t_d_v_s[i] = -1.0
        '''
        t_j[i] = t_a[i] - t_a[i-1]
        if _JERK_MIN > t_j[i]:
            t_j[i] = _JERK_MIN
        if _JERK_MAX < t_j[i]:
            t_j[i] = _JERK_MAX

        t_snap[i] = t_j[i] - t_j[i-1]
        if _SNAP_MIN > t_snap[i]:
            t_snap[i] = _SNAP_MIN
        if _SNAP_MAX < t_j[i]:
            t_snap[i] = _SNAP_MAX

        '''
        if t_s[i] > 0:
            # update tangential acceleration
            t_a[i] = t_s[i] - t_s[i-1]
            cur_dir = np.array([cur_pos[0]-prev_pos[0], cur_pos[1]-prev_pos[1]])
            # find signed angle determined by direction change with respect to x axis and previous position
            # first get abs value of angle
            angle = np.arccos(np.dot(np.array([1.0, 0]), cur_dir)/(1 * np.linalg.norm(cur_dir)))
            if np.isnan(angle):
                if (cur_dir == prev_dir).all():
                    angle = 0.0
                else:
                    angle = np.pi
            angles[i] = np.sign(np.cross(prev_dir, cur_dir)) * angle
            # now update angular speed as change of angle over time unit
            a_s[i] = angles[i] - angles[i-1]
            a_a[i] = a_s[i] - a_s[i-1]

        prev_dir = cur_dir
        '''
        prev_pos = cur_pos

    time = len(trip) - 1
    distance = t_s.sum()

    # print np.histogram(t_d_v_s[t_d_v_s<=0], bins=10)

    t_s_hist, _ = np.histogram(t_s, bins=_SPEED_BINS)
    t_a_hist, _ = np.histogram(t_a, bins=_ACCEL_BINS)
    t_j_hist, _ = np.histogram(t_j, bins=_JERK_BINS)
    t_snap_hist, _ = np.histogram(t_snap, bins=_SNAP_BINS)
    # t_d_v_s_hist_1, _ = np.histogram(t_d_v_s[t_d_v_s>=0], bins=np.linspace(0.0, 1.0, 4))
    # t_d_v_s_hist_2, _ = np.histogram(t_d_v_s[t_d_v_s<=0], bins=np.linspace(-1.0, 0.0, 4))

    # to_p, _ = np.histogram(t_snap, bins=_SNAP_BINS)
    # print to_p
    t_s_hist = t_s_hist / float(t_s_hist.sum())
    t_a_hist = t_a_hist / float(t_a_hist.sum())
    t_j_hist = t_j_hist / float(t_j_hist.sum())
    t_snap_hist = t_snap_hist / float(t_snap_hist.sum())
    #t_d_v_s_hist_1 = t_d_v_s_hist_1 / float(t_d_v_s_hist_1.sum())
    #t_d_v_s_hist_2 = t_d_v_s_hist_2 / float(t_d_v_s_hist_2.sum())

    c_tspeed = np.array([t_s.min(), t_s.max(), t_s.var(), t_s.mean()])
    c_taccel = np.array([t_a.min(), t_a.max(), t_a.var(), t_a.mean()])
    c_tjerk = np.array([t_j.min(), t_j.max(), t_j.var(), t_j.mean()])
    c_tsnap = np.array([t_snap.min(), t_snap.max(), t_snap.var(), t_snap.mean()])
    #c_tdsv = np.array([t_d_v_s.min(), t_d_v_s.max(), t_d_v_s.var(), t_d_v_s.mean()])

    return np.hstack((time, distance, c_tspeed, c_taccel, c_tjerk, c_tsnap, t_s_hist, t_a_hist, t_j_hist,
                      t_snap_hist))


def compute_pdf(samples, vmax, vmin, bins):
    samples[samples > vmax] = vmax
    samples[samples < vmin] = vmin

    s_cdf, _ = np.histogram(samples, bins=bins, density=True)
    if np.isnan(s_cdf).any():
        s_cdf = s_cdf.astype(float)
        s_cdf = np.ones(len(bins)-1)
    # normalize so they are probabilities
    s_cdf /= np.sum(s_cdf)
    return s_cdf


def get_info_for_driver(driver):
    samples = np.zeros([200, _N_F])
    for i in range(1, 201):
        samples[i-1, :] = get_info_for_trip("data/drivers/%s/%s.csv" % (driver, i))
    return samples

def clean_noise(X, to_keep):
    """
    X is (ntrips x n_feat), to_keep is # of sample to save
    """
    k_means = sklearn.cluster.KMeans(n_clusters=1)

    A = np.copy(X)

    X_noise = np.zeros([A.shape[0]-to_keep, A.shape[1]])
    
    k_means.fit(A)

    for i in range(X.shape[0]-to_keep):
        k_means.fit(A)
        d = k_means.transform(A)[:, 0]
        ind = np.argsort(d)[::-1][:1]
        n_ind = np.array(list(set(range(len(d)))-set(ind)))
        X_noise[i] = np.copy(A[ind])
        A = np.copy(A[n_ind]) 
    
    return {
        "X_clean": A,
        "X_noise": X_noise
    }


def load_driver(driver):
    data = np.loadtxt("data/features/%s.csv" % driver)
    target = np.ones(200)*int(driver)
    return data, target

def load_data(drivers=None):
    folders = os.listdir("data/drivers/")
    
    if not drivers:
        drivers = filter(lambda idd: idd[0] != ".", folders)
    
    data = np.zeros([len(drivers) * 200, _N_F])
    target = np.zeros(len(drivers) * 200)
    for i, d in enumerate(drivers):
        X, y = load_driver(d)
        data[i*200:(i+1)*200, :] = X
        target[i*200:(i+1)*200] = y
        
    return data, target


def load_outliers_pca(driver, o_t_count, avail_drivers=None):
    if not avail_drivers:
        folders = os.listdir("data/drivers/")
        avail_drivers = filter(lambda idd: idd[0] != ".", folders)
    avail_drivers.remove(driver)
    avail_drivers = sklearn.utils.shuffle(avail_drivers)
    outliers = avail_drivers[0:o_t_count]
    
    data = np.zeros([len(outliers), _COMP_PCA])
    target = np.zeros(len(outliers)).astype(int)

    for i, o in enumerate(outliers):
        d, _ = load_driver_pca(o)
        data[i] = d[random.randint(0, 199)]
    return data, target

def load_driver_pca(driver):
    data = np.loadtxt("data/reduced/%s.csv" % driver)
    target = (np.ones(200)*int(driver)).astype(int)
    return data, target










