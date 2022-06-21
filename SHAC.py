import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import qmc

def SHAC(function, 
         bounds, 
         init='lhs', 
         args=None, 
         popsize=20, 
         maxiter=15, 
         max_clfs=10, 
         verbose=False, 
         callback=None):
    
    with SHACSolver(function=function,
                    bounds=bounds,
                    init=init,
                    args=args,
                    popsize=popsize,
                    maxiter=maxiter,
                    max_clfs=max_clfs,
                    verbose=verbose,
                    callback=callback) as solver:
         
        best_candidate, best_candidate_energy = solver.optimize()
    
    ret = {'x': best_candidate,
           'fun': best_candidate_energy,
           'nfev': solver.nfev}

    return ret

class SHACSolver:
    def __init__(self, 
                 function, 
                 bounds, 
                 init='lhs', 
                 args=None, 
                 popsize=20, 
                 maxiter=15, 
                 max_clfs=10, 
                 verbose=False, 
                 callback=None):
        
        assert callable(function), "function must be a callable function"
        try:
            iter(bounds)
            assert len(bounds[0]) == 2, "length of each bound needs to be 2, i.e. [lb, ub]"
        except TypeError:
            raise AssertionError("bounds needs to be iterable, e.g. [[lb, ub], [lb, ub]...]")
        if type(init) == str:
            assert init in ('lhs', 'sobol', 'halton'), "init should be one of the following: lhs, sobol, halton"
        elif type(init) == np.ndarray:
            assert len(init) == len(bounds), "init candidates does not match bounds dimensions"
        else:
            raise AssertionError("init should be one of the following: lhs, sobol, halton, or an numpy array")
        assert type(popsize) == int and popsize != 0, "popsize needs to be a non-zero integer"
        assert type(maxiter) == int and maxiter != 0, "maxiter needs to be a non-zero integer"
        assert type(max_clfs) == int and max_clfs != 0, "max_clfs needs to be a non-zero integer"
        assert type(verbose) == bool, "verbose option needs to be boolean"
        assert callback is None or callable(callback), "callback must be a callable function or None"
        
        self.function = self._function_wrapper(function, args)
        self.bounds = bounds
        self.maxiter = maxiter
        self.max_clfs = max_clfs
        self.clfs = []
        self.dim = len(bounds)
        self.verbose = verbose
        self.callback = callback
        
        # initialize populaiton
        if type(init) == str:
            if init == 'lhs':
                self.population = qmc.LatinHypercube(self.dim).random(n=popsize)
            elif init == 'sobol':
                self.population = qmc.Sobol(self.dim).random(n=popsize)
            elif init == 'halton':
                self.population = qmc.Halton(self.dim).random(n=popsize)
            self.popsize = len(self.population)
        else:
            self.population = init
            self.popsize = len(init)
        
        self.population_energies = self.function(self._scale_parameters(self.population)).reshape(-1)
        best_ind = np.argmin(self.population_energies)
        self.best_candidate = self.population[best_ind]
        self.best_candidate_energy = self.population_energies[best_ind]
        self.nfev = self.popsize
        
        
    def __enter__(self):
        return self


    def __exit__(self, exception_type, exception_value, traceback):
        return self

    
    def _function_wrapper(self, function, args):
        self.args = [] if args is None else args
        def wrapperd_function(x): return function(x, *self.args)
        return wrapperd_function
        
    
    def _scale_parameters(self, x):
        x = np.atleast_2d(x)
        return qmc.scale(x, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
    
    
    def _get_training_y(self, y):
        median = np.median(y)
        return np.where(y > median, 1, 0)
    
    
    def _get_new_points(self):
        points_generated = 0
        sample_size = self.popsize
        while points_generated < self.popsize:
            trials = qmc.LatinHypercube(self.dim).random(n=sample_size)
            passed = np.ones(sample_size).astype(bool)
            
            # run through each of classifier
            for clf in self.clfs:
                passed = np.logical_and(passed, clf.predict(trials).astype(bool))
                
            p = np.sum(passed)
            if p + points_generated > self.popsize:
                self.population[points_generated:, :] = trials[passed, :][:self.popsize - points_generated]
                break
            else:
                self.population[points_generated : points_generated+p, :] = trials[passed, :]
            
            points_generated += p    
            sample_size = int((self.popsize - points_generated)/(p+1) * sample_size)
            
            if self.verbose:
                print(f"generating new candidates, {points_generated} completed...")
        self.population_energies = self.function(self._scale_parameters(self.population)).reshape(-1)
        self.nfev += self.popsize
            
   
    def optimize(self):
        for i in range(self.maxiter):
            if self.verbose:
                print(f"iteration {i}, current best: {self.best_candidate_energy}")
            # evaluate population on function and train random forest model
            if len(self.clfs) < self.max_clfs:
                y_train = self._get_training_y(self.population_energies)
                self.clfs.append(RandomForestClassifier().fit(self.population, y_train))
            
            # generate new population and update best solution
            self._get_new_points()
            
            best_ind = np.argmin(self.population_energies)
            if self.population_energies[best_ind] < self.best_candidate_energy:
                self.best_candidate_energy = self.population_energies[best_ind]
                self.best_candidate = self.population[best_ind]
            
            if self.callback is not None:
                if self.callback(self.best_candidate):
                    break
        
        return self.best_candidate, self.best_candidate_energy
                
            
            
            
            
            
            
            
            
            
            
            
            
            
            