# SHAC
Successive Halving and Classification (SHAC)

Implementation of "Successive Halving and Classification" optimization algorithm, following Scipy optimize syntax

    # optimize function within bounds:
    result = SHAC(function, bounds, init='lhs', args=None, popsize=20, maxiter=15, max_clfs=10, verbose=False, callback=None)

    # view results:
    result.x     # optimization solution
    result.fun   # function value at solution
    result.nfev  # number of function evaluations/calls made

Arguments
- function, the objective function to be minimized. Must be in the form f(x, *args), where x is the argument in the form of a 1-D array and args is a tuple of any additional fixed parameters needed to completely specify the function.
- bounds, bounds for variables. Iterable of length equal to dimension of x, with each element a tuple or list of (lower bound, upper bound) for the respective dimension. I.e. [[lb, ub], [lb, ub]...]
- init, initial points generation method, one of either 'lhs', 'sobol', 'halton'. Default is 'lhs' (latin hypercube)
- args, tuple of any additional fixed parameters needed to completely specify the objective function. Default is None
- popsize, number of samples tested each iteration. Default is 20
- maxiter, number of iterations or successive halving before termination. Default is 15 
- max_clfs=10, max number of classifiers to be trained. Default is 10
- verbose, whether to print information during optimization process. Default is False
- callback, function in form callback(xk), where xk is the best solution for each iteration. Default is None

# Reference
https://arxiv.org/abs/1805.10255
