import autograd.numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
from utils.stochastic_gradient import StochasticGradientDescent
import torch

def riemannian_cpd_spd(manifold, X, lambda_0, lambda_1):
    # optimizer
    optimizer0 = StochasticGradientDescent(step_size = lambda_0, num_iter = 1)
    optimizer1 = StochasticGradientDescent(step_size = lambda_1, num_iter = 1)
    # online CPD on Riemannian manifolds
    stat = []
    for matrix in X:
        @pymanopt.function.pytorch(manifold)
        def cost(point):
            temp1 = torch.linalg.eig(torch.from_numpy(matrix))
            temp2 = temp1[0].real
            c = temp1[1].real @ torch.diag(torch.sqrt(1/(torch.where(temp2 > 0, temp2, torch.tensor(1e-6, dtype=torch.float64))))) @ temp1[1].real.T
            temp3 = c @ point @ c
            temp4 = torch.linalg.eig(temp3)[0].real
            temp5 = torch.log(torch.where(temp4 > 0, temp4, torch.tensor(1e-6, dtype=torch.float64)))
            return torch.norm(temp5)**2
        problem = pymanopt.Problem(manifold, cost)
        if np.all(matrix == X[0]):
            result0 = optimizer0.run(problem, initial_point=matrix)
            result1 = optimizer1.run(problem, initial_point=matrix)
        else:
            result0 = optimizer0.run(problem, initial_point=result0.point)
            result1 = optimizer1.run(problem, initial_point=result1.point)
        stat.append(manifold.dist(result0.point, result1.point))
    return stat

def riemannian_cpd_grassmann(manifold, X, lambda_0, lambda_1):
    # optimizer
    optimizer0 = StochasticGradientDescent(step_size = lambda_0, num_iter = 1)
    optimizer1 = StochasticGradientDescent(step_size = lambda_1, num_iter = 1)
    # online CPD on Riemannian manifolds
    stat = []
    for matrix in X:
        @pymanopt.function.pytorch(manifold)
        def cost(point):
            temp1 = torch.from_numpy(matrix.transpose()) @ point
            temp2 = torch.linalg.svd(temp1)[1]
            temp3 = torch.acos(torch.clamp(temp2, -1+1e-6, 1-1e-6))
            return torch.norm(temp3)**2
        problem = pymanopt.Problem(manifold, cost)
        if np.all(matrix == X[0]):
            result0 = optimizer0.run(problem, initial_point=matrix)
            result1 = optimizer1.run(problem, initial_point=matrix)
        else:
            result0 = optimizer0.run(problem, initial_point=result0.point)
            result1 = optimizer1.run(problem, initial_point=result1.point)
        stat.append(manifold.dist(result0.point, result1.point))
    return stat
