# coding: utf-8
# Script for validating the Gaussian distribution
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

from utils.draw_figure import makedir, normal
from utils.riemannian_cpd import riemannian_cpd_spd
from utils.functions import generate_random_SPD_mtx, generate_random_SPD_Wishart
import seaborn as sns

figure_path = './figures/'
# parameter settings
lambda_0 = 1e-2
lambda_1 = 2e-2

# experiment setups
start_point = 200
T = start_point + 1e3
N = 8 # Dimension of the space

# generate parameters for two Wishart distributions
np.random.seed(1)
temp = np.random.randn(N,N)
eigsv = np.random.rand(N) + 1e-6 # positive
eigsv_v = 1.6 * np.random.rand(1)
M0 = generate_random_SPD_mtx(temp, eigsv)

# define manifold
manifold = pymanopt.manifolds.positive_definite.SymmetricPositiveDefinite(N)

stat_all = []
X = []
for t in range(int(T)):
    X.append(generate_random_SPD_Wishart(N+3, M0))
stat_all = riemannian_cpd_spd(manifold, X, lambda_0, lambda_1)

# draw figures
data = np.array(stat_all[start_point:])
if not os.path.exists(figure_path):
    makedir(figure_path)
fig = plt.figure(figsize = (4, 2.5), dpi = 150)
sns.set_theme(style="white", palette=None)
ax = sns.histplot(x=data, stat="density", color = "#1F77B4", alpha=0.4)
normal(data.mean(), data.std(), color="#2F7FC1", linewidth=2)
plt.tight_layout()
plt.savefig(figure_path + "distribution.pdf", bbox_inches='tight')
plt.show()