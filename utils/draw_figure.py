import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

def comp_roc(stat_all, change_location, N_th=100, start_point=0):
    TH = np.linspace(np.min(np.array(stat_all)[:, start_point:]), 1.1*np.max(np.array(stat_all)[:, start_point:]), N_th)
    pfa = np.array([sum(np.any(stat[start_point:change_location] >= threshold) for stat in stat_all) for threshold in TH]) / len(stat_all)
    pd = np.array([sum(np.any(stat[change_location:] >= threshold) for stat in stat_all) for threshold in TH]) / len(stat_all)
    return pfa, pd

def comp_arl_mdd(stat_all, change_location, N_th=100, start_point=0):
    TH = np.linspace(np.min(np.array(stat_all)[:, start_point:]), np.max(np.array(stat_all)[:, start_point:]), N_th)
    arl = []
    mdd = []
    for threshold in TH:
        arl_stat = []
        mdd_stat = []
        for stat in stat_all:
            arl_temp = np.argwhere(stat[start_point:change_location] >= threshold)
            mdd_temp = np.argwhere(stat[change_location:] >= threshold)
            if np.size(arl_temp) != 0:
                arl_stat.append(arl_temp[0])
            else:
                arl_stat.append(change_location - start_point)
            if np.size(mdd_temp) != 0:
                mdd_stat.append(mdd_temp[0])
            else:
                mdd_stat.append(change_location - start_point)
        arl.append(sum(arl_stat) / float(len(arl_stat)))
        mdd.append(sum(mdd_stat) / float(len(mdd_stat)))
    arl = np.array(arl, dtype=object)
    mdd = np.array(mdd, dtype=object)
    return arl, mdd

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def normal(mean, std, color, linewidth):
    x = np.linspace(mean-4*std, mean+4*std, 200)
    p = norm.pdf(x, mean, std)
    z = plt.plot(x, p, color, linewidth=linewidth)