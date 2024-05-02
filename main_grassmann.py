# coding: utf-8
# Script for performing change point detection on Grassmann manifolds
#
# Reference: 
# Non-parametric Online Change Point Detection on Riemannian Manifolds
# Xiuheng Wang, Ricardo Borsoi, Cédric Richard
#
# 2022/11
# Implemented by
# Xiuheng Wang, Ricardo Borsoi
# xiuheng.wang@oca.eu, raborsoi@gmail.com

import pymanopt
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from matplotlib.pyplot import MultipleLocator
import seaborn as sns

from utils.baselines import frechet_cpd
import utils.onlinecp as ocp
from utils.node import node
from utils.draw_figure import comp_roc, comp_arl_mdd, makedir
from utils.riemannian_cpd import riemannian_cpd_grassmann
from utils.functions import generate_random_SPD_mtx, generate_random_mtx_normal

figure_path = './figures/'
# parameter settings
lambda_0 = 1e-2
lambda_1 = 2e-2
# F-CPD
len_win = 64
# NODE
layers=[32]*3
# Scan-B
B = 50
N_window = 3
# NEWMA
c = 2
lambda_0_newma = (c**(1/B)-1)/(c**((B+1)/B)-1)
lambda_1_newma = c*lambda_0_newma

# experiment setups
T = 2000
Tc = 1500
N = 20 # Dimension of the ambient space.
P = 5 # Dimension of the subspaces.
Iter = 1e4

# generate parameters for two matrix normal distributions
np.random.seed(1)
temp = np.random.randn(N,N)
eigsv = np.random.rand(N) + 1e-6 # positive
U = generate_random_SPD_mtx(temp, eigsv)
temp = np.random.randn(N,N)
eigsv = np.random.rand(N) + 1e-6 # positive
V = generate_random_SPD_mtx(temp, eigsv)
M0 = np.random.randn(N,N) + 1
M1 = M0 + 0.03 * np.random.randn(N,N)

# define manifold
manifold = pymanopt.manifolds.grassmann.Grassmann(N, P)

stat_all = []
stat_frechet_all = []
stat_node_all = []
stat_scanb_all = []
stat_newma_all = []
d = int(N*P)
W = np.random.randn(2000, d)/np.sqrt(d)
for _ in tqdm(range(int(Iter))):
    X = []
    for t in range(T):
        if t < Tc:
            temp = generate_random_mtx_normal(M0, U, V)
            X.append(np.linalg.svd(temp)[0][:, :P]) # Truncated SVD 
        else:
            temp = generate_random_mtx_normal(M1, U, V)
            X.append(np.linalg.svd(temp)[0][:, :P]) # Truncated SVD
    X_vec = [item.flatten() for item in X] 
    stat_frechet_all.append(frechet_cpd(X, len_win))
    stat_node_all.append(node(np.array(X_vec), len_win, layers))
    ocp_object = ocp.ScanB(d, store_result=True, B=B, N=N_window,
                            kernel_func=lambda x, y: ocp.gauss_kernel(x, y, d))
    ocp_object.apply_to_data(np.array(X_vec))
    stat_scanb_all.append(np.array(ocp_object.dist))
    ocp_object = ocp.Newma(store_result=True, updt_coeff=lambda_0_newma, updt_coeff2=lambda_1_newma,
                            updt_func=lambda x: ocp.fourier_feature(x, W))
    ocp_object.apply_to_data(np.array(X_vec))
    stat_newma_all.append(np.array(ocp_object.dist))
    stat_all.append(riemannian_cpd_grassmann(manifold, X, lambda_0, lambda_1))

# gather all test statistics
stats = []
stats.append(stat_frechet_all)
stats.append(stat_node_all)
stats.append(stat_scanb_all)
stats.append(stat_newma_all)
stats.append(stat_all)

# set names and colors
names = ["F-CPD", "NODE", "Scan-B", "NEWMA", "Our"]
colors = ["#BEB8DC", "#82B0D2", "#8ECFC9", "#FFBE7A", "#FA7F6F"]

# draw figures
start_point = 400
if not os.path.exists(figure_path):
    makedir(figure_path)
fig = plt.figure(figsize = (6, 6), dpi = 120)
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
plt.savefig(figure_path + "simulation_grassmann.pdf", bbox_inches='tight')

N_th = 1000
fig = plt.figure(figsize = (3.2, 3.0), dpi = 150)
for index in range(len(names)):
    pfa, pd = comp_roc(stats[index], Tc, N_th, start_point)
    plt.plot(pfa, pd, color=colors[index], label=names[index])
plt.xlabel("False alarm rate")
plt.ylabel("Detection rate")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(figure_path + "roc_grassmann.pdf", bbox_inches='tight')

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
plt.savefig(figure_path + "arl_mdd_grassmann.pdf", bbox_inches='tight')

fig = plt.figure(figsize = (6, 7), dpi = 120)
sns.set_theme(style="white", palette=None)
for index in range(len(names)):
    ax = fig.add_subplot(len(names), 1, index+1)
    stats_all = np.array(stats[index])
    stats_null = stats_all[:, int((Tc+start_point)/2.0)]
    stats_mean = np.mean(stat_all, axis=0)[Tc:]
    stats_peak = stats_all[:, Tc + np.argmax(stats_mean)]
    ax = sns.histplot(x=stats_null, stat="count", color = "#1F77B4", alpha=0.4)
    ax = sns.histplot(x=stats_peak, stat="count", color = "#FA7F6F", alpha=0.4)
    plt.title(names[index], x=0.93, y=0.72)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.subplots_adjust(hspace = 0.15)
plt.savefig(figure_path + "histogram_grassmann.pdf", bbox_inches='tight')

plt.show()