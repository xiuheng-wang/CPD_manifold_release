# coding: utf-8
# Script for performing change point detection with Fréchet CPD.
#
# 2023/01
# Implemented by
# Xiuheng Wang
# xiuheng.wang@oca.eu

import numpy as np

def median_trick(x):
  n = np.shape(x)[1]
  dists = []
  for j in range(n):
      for i in range(j-1):
          dists.append(np.linalg.norm(x[:, j] - x[:, i]))
  return np.median(dists)**2

def frechet_stat(x):
    """ A function to compute the Fr ́echet statistics for change point detection as used in:
    Dubey and H.-G. M ̈uller, “Fréchet change-point detection,”
    The Annals of Statistics, vol. 48, no. 6, pp. 3312–3335, 2020.
    Usage: test = frechet_stat(x, c)
    Inputs:
        * x: a list of manifold-valued variables.
        * c: a parameter to control the range of computing test statistic
    Outputs:
        * test: a list of test statistics."""
        
    nt = len(x)
    m = np.mean(x, axis = 0)
    t = int(np.floor(nt*0.5))
    u = 0.5
    m0 = np.mean(x[:t], axis = 0)
    m1 = np.mean(x[t:], axis = 0)
    V0 = np.mean(np.sum((x[:t] - m0)**2, axis = (1,2)))
    V1 = np.mean(np.sum((x[t:] - m1)**2, axis = (1,2)))
    V0c = np.mean(np.sum((x[:t] - m1)**2, axis = (1,2)))
    V1c = np.mean(np.sum((x[t:] - m0)**2, axis = (1,2)))
    dsq = np.sum((x - m)**2, axis = (1,2))
    sigma = np.mean(dsq**2) - np.mean(dsq)**2
    add_factor = (V0c-V0) + (V1c-V1)
    stat = u*(1-u)*(((V0-V1)**2) + add_factor**2) / sigma
    return stat

def frechet_cpd(data, swl=64):
    nt = len(data)
    test = np.zeros(nt)
    for i in range(2*swl, nt):
        data_segment = data[i - 2*swl : i]
        test[i] = frechet_stat(data_segment)
    return test