# coding: utf-8
# Script for performing adaptive threshold selection
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

from utils.draw_figure import makedir
from utils.riemannian_cpd import riemannian_cpd_spd
from utils.functions import generate_random_SPD_mtx, generate_random_SPD_Wishart

figure_path = './figures/'
# parameter settings
lambda_0 = 1e-2
lambda_1 = 2e-2

# experiment setups
T = 400
Tc = 200
N = 8 # Dimension of the space
Iter = 8

# generate parameters for two Wishart distributions
np.random.seed(1)
temp = np.random.randn(N,N)
eigsv = np.random.rand(N) + 1e-6 # positive
eigsv_v = 1.6 * np.random.rand(1)
M0 = generate_random_SPD_mtx(temp, eigsv)
M1 = generate_random_SPD_mtx(temp, eigsv + eigsv_v)

# define manifold
manifold = pymanopt.manifolds.positive_definite.SymmetricPositiveDefinite(N)

stat_all = []
X = []
for _ in tqdm(range(int(Iter))):
    for t in range(T):
        if t < Tc:
            X.append(generate_random_SPD_Wishart(N+3, M0))
        else:
            X.append(generate_random_SPD_Wishart(N+3, M1))
stat_all = riemannian_cpd_spd(manifold, X, lambda_0, lambda_1)

mean = 0
variance = 0
alpha = 0.005
a = 1.64
threshold = []
for stat in stat_all:
    mean = (1-alpha) * mean + alpha*stat
    variance = (1-alpha) * variance + alpha*stat**2
    sigma = np.sqrt(variance - mean**2)
    threshold.append(mean+a*sigma)

# draw figures
start_point = 6*T - Tc
if not os.path.exists(figure_path):
    makedir(figure_path)
fig = plt.figure(figsize = (4, 2.5), dpi = 150)
plt.plot(stat_all, color = "#2F7FC1")
plt.plot(threshold, color = "#FA7F6F")
plt.legend(["Our", "Adapt. thres."], loc="upper left")
for i in range(2*Iter):
    plt.axvline(Tc*(i+1), color = "#999999", linestyle = "--")
ax = plt.gca()
y_major_locator = plt.MaxNLocator(5)
ax.yaxis.set_major_locator(y_major_locator)
plt.xlim(start_point, T*Iter)
plt.ylim(0.09, 0.53)
plt.tight_layout()
plt.savefig(figure_path + "adaptive_threshold.pdf", bbox_inches='tight')
plt.show()