# coding: utf-8
# Script for performing change point detection in voice activity detection
#
# Reference: 
# Non-parametric Online Change Point Detection on Riemannian Manifolds
# Xiuheng Wang, Ricardo Borsoi, Cédric Richard
#
# 2023/12
# Implemented by
# Xiuheng Wang
# xiuheng.wang@oca.eu

import pymanopt
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from matplotlib.pyplot import MultipleLocator

from utils.baselines import median_trick
import utils.onlinecp as ocp
from utils.draw_figure import comp_roc, comp_arl_mdd, makedir
from utils.riemannian_cpd import riemannian_cpd_spd, riemannian_cpd_grassmann
from utils.functions import import_vad_data

# parameter settings
lambda_0_spd = 1e-2
lambda_1_spd = 2e-2
lambda_0_sub = 1e-2
lambda_1_sub = 2e-2
# Scan-B
B = 50
N_window = 3
# NEWMA
c = 2
lambda_0_newma = (c**(1/B)-1)/(c**((B+1)/B)-1)
lambda_1_newma = c*lambda_0_newma

# paths of data and figures
root_path = "..\\data\\"
figure_path = './figures/'
if not os.path.exists(figure_path):
    makedir(figure_path)

# experiment setups
nb_change = 1e4
length_noise = 15
length_speech = 4
SNR = 0.5  # 0: only noise, 1: only speech
nperseg = 128*2
sample_factor = 8
no_show = 1
print("SNR:", 10*np.log10(SNR))
X, X_full = import_vad_data(root_path, nb_change, length_noise, length_speech, SNR, nperseg, sample_factor, no_show)
window_length = 32

# define manifold
N = np.shape(X)[-1] # dimensionality of SPD
manifold_cov = pymanopt.manifolds.positive_definite.SymmetricPositiveDefinite(N)
P = 1
manifold_sub = pymanopt.manifolds.grassmann.Grassmann(N, P)

# compute covariance matrices and subspaces
print("Compute features:")
X_cov = []
X_sub = []
for x in tqdm(X):
    i = window_length
    x_cov = []
    x_sub = []
    while i <= np.shape(x)[0]:
        samples = x[i-window_length: i]
        covariance = np.cov(samples.T)
        x_cov.append(covariance)
        samples -= samples.mean(axis=0)
        subspace = np.linalg.svd(samples / np.sqrt(N*window_length))[2][:P, :].T
        x_sub.append(subspace)
        i += 1
    X_cov.append(x_cov)
    X_sub.append(x_sub)

print("Detect change points:")
stat_scanb_all = []
stat_newma_all = []
stat_spd_all = []
stat_sub_all = []
d = np.size(X_full[0][0])
sigma = median_trick(np.transpose(X_full[0]))
W = np.random.randn(2000, d)/np.sqrt(sigma)
for index in tqdm(range(int(nb_change))):
    # baselines
    ocp_object = ocp.ScanB(d, store_result=True, B=B, N=N_window,
                            kernel_func=lambda x, y: ocp.gauss_kernel(x, y, sigma))
    ocp_object.apply_to_data(np.array(X_full[index]))
    stat_scanb_all.append(np.array(ocp_object.dist)[window_length-1:])
    ocp_object = ocp.Newma(store_result=True, updt_coeff=lambda_0_newma, updt_coeff2=lambda_1_newma,
                            updt_func=lambda x: ocp.fourier_feature(x, W))
    ocp_object.apply_to_data(np.array(X_full[index]))
    stat_newma_all.append(np.array(ocp_object.dist)[window_length-1:])
    stat_spd_all.append(riemannian_cpd_spd(manifold_cov, X_cov[index], lambda_0_spd, lambda_1_spd))
    stat_sub_all.append(riemannian_cpd_grassmann(manifold_sub, X_sub[index], lambda_0_sub, lambda_1_sub))

# gather all test statistics
stats = []
stats.append(stat_scanb_all)
stats.append(stat_newma_all)
stats.append(stat_sub_all)
stats.append(stat_spd_all)

# set names and colors
names = ["Scan-B", "NEWMA", "Our-sub", "Our-cov"]
colors = ["#8ECFC9", "#FFBE7A", "#FFC0B9", "#FA7F6F"]

# draw figures
T = np.shape(X)[1]
Tc = int(T * (length_noise - length_speech) / length_noise) - window_length + 1
T -=  window_length - 1
start_point = 300
fig = plt.figure(figsize = (6, 5), dpi = 120)
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
plt.savefig(figure_path + "vad.pdf", bbox_inches='tight')

N_th = 1000
fig = plt.figure(figsize = (3.2, 3.0), dpi = 150)
for index in range(len(names)):
    pfa, pd = comp_roc(stats[index], Tc, N_th, start_point)
    plt.plot(pfa, pd, color=colors[index], label=names[index])
plt.xlabel("False alarm rate")
plt.ylabel("Detection rate")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(figure_path + "roc_vad.pdf", bbox_inches='tight')

fig = plt.figure(figsize = (3.2, 3.0), dpi = 150)
for index in range(len(names)):
    arl, mdd = comp_arl_mdd(stats[index], Tc, N_th, start_point)
    plt.plot(arl, mdd, color=colors[index], label=names[index])
plt.xlim(0, 1000)
plt.ylim(0, 50)
y_major_locator = MultipleLocator(10)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.xlabel("Average run length")
plt.ylabel("Mean detection delay")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(figure_path + "arl_mdd_vad.pdf", bbox_inches='tight')

plt.show()
