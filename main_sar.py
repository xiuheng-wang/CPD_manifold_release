# coding: utf-8
# Script for performing change point detection in skeleton-based action recognition
#
# Reference: 
# Non-parametric Online Change Point Detection on Riemannian Manifolds
# Xiuheng Wang, Ricardo Borsoi, Cédric Richard
#
# 2024/03
# Implemented by
# Xiuheng Wang
# xiuheng.wang@oca.eu

import pymanopt
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from matplotlib.pyplot import MultipleLocator
from scipy.io import loadmat
import random

from utils.baselines import median_trick
import utils.onlinecp as ocp
from utils.draw_figure import comp_roc, comp_arl_mdd, makedir
from utils.riemannian_cpd import riemannian_cpd_spd, riemannian_cpd_grassmann
from utils.functions import import_vad_data

# parameter settings
lambda_0_spd = 2.5e-2
lambda_1_spd = 5e-2
# Scan-B
B = 12
N_window = 3
# NEWMA
c = 2
lambda_0_newma = (c**(1/B)-1)/(c**((B+1)/B)-1)
lambda_1_newma = c*lambda_0_newma

# paths of data and figures
spd_path = "..\\data\\HDM05_SPDData\\feature"
figure_path = './figures/'
if not os.path.exists(figure_path):
    makedir(figure_path)

# experiment setups
nb_change = 1e3
nb_samples = 200

# define manifold
N = 93 # dimensionality of SPD
manifold_cov = pymanopt.manifolds.positive_definite.SymmetricPositiveDefinite(N)

# list of spd and subspace matrices
spd_all = []
dir_names = os.listdir(spd_path)
for dir_name in dir_names:
    file_names = os.listdir(spd_path + "\\" + dir_name)
    spd_items = []
    subspace_items = []
    if len(file_names) >= nb_samples:
        for file_name in file_names:
            spd = loadmat(spd_path + "\\" + dir_name + "\\" + file_name)['Y1']
            spd_items.append(spd.astype(np.double))
        random.shuffle(spd_items)
        spd_all.append(spd_items[:nb_samples])

print("Detect change points:")
stat_scanb_all = []
stat_newma_all = []
stat_spd_all = []
X_cov = spd_all[0] + spd_all[1]
X_vec = [item[np.triu_indices(N)] for item in X_cov]
sigma = median_trick(np.transpose(np.array(X_vec)))
d = np.size(X_vec[0])
W = np.random.randn(2000, d)/np.sqrt(sigma)
for _ in tqdm(range(int(nb_change))):
    random.shuffle(spd_all)
    category_1 = spd_all[0]
    category_2 = spd_all[1]
    random.shuffle(category_1)
    random.shuffle(category_2)
    X_cov = category_1 + category_2
    X_vec = [item[np.triu_indices(N)] for item in X_cov]
    # baselines
    ocp_object = ocp.ScanB(d, store_result=True, B=B, N=N_window,
                            kernel_func=lambda x, y: ocp.gauss_kernel(x, y, sigma))
    ocp_object.apply_to_data(np.array(X_vec))
    stat_scanb_all.append(np.array(ocp_object.dist))
    ocp_object = ocp.Newma(store_result=True, updt_coeff=lambda_0_newma, updt_coeff2=lambda_1_newma,
                            updt_func=lambda x: ocp.fourier_feature(x, W))
    ocp_object.apply_to_data(np.array(X_vec))
    stat_newma_all.append(np.array(ocp_object.dist))
    stat_spd_all.append(riemannian_cpd_spd(manifold_cov, X_cov, lambda_0_spd, lambda_1_spd))

# gather all test statistics
stats = []
stats.append(stat_scanb_all)
stats.append(stat_newma_all)
stats.append(stat_spd_all)

# set names and colors
names = ["Scan-B", "NEWMA", "Our"]
colors = ["#8ECFC9", "#FFBE7A", "#FA7F6F"]

# draw figures
T = 2*nb_samples
Tc = nb_samples
start_point = 100
fig = plt.figure(figsize = (6, 4.5), dpi = 120)
for index in range(len(names)):
    ax = fig.add_subplot(len(names), 1, index+1)
    avg = np.mean(stats[index], axis = 0)
    std = np.std(stats[index], axis = 0)
    r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))
    r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))
    ax.plot(range(0, T), avg, color = "#2F7FC1")
    ax.fill_between(range(0, T), r1, r2, alpha=0.2)
    plt.axvline(Tc, color = "#FA7F6F")
    plt.legend([names[index]], loc = 1)
    plt.xlim(start_point, T)
    plt.ylim(0.9*np.min(r1[start_point:]), 1.1*np.max(r2[start_point:]))
plt.tight_layout()
plt.subplots_adjust(hspace = 0.28)
plt.savefig(figure_path + "sar.pdf", bbox_inches='tight')

N_th = 1000
fig = plt.figure(figsize = (3.2, 3.0), dpi = 150)
for index in range(len(names)):
    pfa, pd = comp_roc(stats[index], Tc, N_th, start_point)
    plt.plot(pfa, pd, color=colors[index], label=names[index])
plt.xlabel("False alarm rate")
plt.ylabel("Detection rate")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(figure_path + "roc_sar.pdf", bbox_inches='tight')

fig = plt.figure(figsize = (3.2, 3.0), dpi = 150)
for index in range(len(names)):
    arl, mdd = comp_arl_mdd(stats[index], Tc, N_th, start_point)
    plt.plot(arl, mdd, color=colors[index], label=names[index])
plt.xlim(0, 100)
plt.ylim(0, 5)
y_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.xlabel("Average run length")
plt.ylabel("Mean detection delay")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(figure_path + "arl_mdd_sar.pdf", bbox_inches='tight')

plt.show()
