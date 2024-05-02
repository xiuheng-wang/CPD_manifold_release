import time
from copy import deepcopy

import numpy as np

from pymanopt.optimizers.optimizer import Optimizer, OptimizerResult
from pymanopt.tools import printer


class StochasticGradientDescent(Optimizer):
    """Riemannian stochastic gradient descent algorithm.

    Perform optimization using stochastic gradient descent with a step size.
    This method first computes the partial derivative of the objective, and then
    optimizes by moving in the direction of steepest descent (which is the
    opposite direction to the gradient).

    Args:
        step_size: The step size.
    """

    def __init__(self, step_size=None, num_iter=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if step_size is None:
            self._step_size = 1e-3
        else:
            self._step_size = step_size
        self._num_iter = num_iter 

    # Function to solve optimisation problem using steepest descent.
    def run(
        self, problem, *, initial_point=None
    ) -> OptimizerResult:
        """Run one step of the stochastic gradient descent algorithm.

        Args:
            problem: Pymanopt problem class instance exposing the cost function
                and the manifold to optimize over.
                The class must either
            initial_point: Initial point on the manifold.
                If no value is provided then a starting point will be randomly
                generated.

        Returns:
            Local minimum of the cost function, or the most recent iterate if
            algorithm terminated before convergence.
        """
        manifold = problem.manifold
        # objective = problem.cost
        gradient = problem.riemannian_gradient

        # If no starting point is specified, generate one at random.
        if initial_point is None:
            x = manifold.random_point()
        else:
            x = initial_point

        # if self._verbosity >= 1:
        #     print("Optimizing...")
        # if self._verbosity >= 2:
        #     iteration_format_length = int(np.log10(self._max_iterations)) + 1
        #     column_printer = printer.ColumnPrinter(
        #         columns=[
        #             ("Iteration", f"{iteration_format_length}d"),
        #             ("Cost", "+.16e"),
        #         ]
        #     )
        # else:
        #     column_printer = printer.VoidPrinter()

        # column_printer.print_header()

        # Initialize iteration counter and timer
        start_time = time.time()

        for iteration in range(self._num_iter):
        # Calculate new cost, grad
            # cost = objective(x)
            grad = gradient(x)

            # column_printer.print_row([iteration, cost])

            # Descent direction is minus the gradient
            desc_dir = -grad

            # Perform stochastic gradient descent
            x = manifold.retraction(x, self._step_size*desc_dir)

        return self._return_result(
            start_time=start_time,
            point=x,
            cost=None,
            iterations=iteration,
            stopping_criterion=None,
            step_size=self._step_size
        )
