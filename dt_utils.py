"""Various code snippets that I find myself commonly using across projects
"""

import timeit
import numpy as np
import matplotlib.pyplot as plt
import joblib
        
class Timer: 
    """Idea from: http://preshing.com/20110924/timing-your-code-using-pythons-with-statement/
    """
    def __init__(self, name='Timer', verbose=True):
        self.name = name
        self.verbose = verbose
    
    def __enter__(self):  
        if self.verbose:
            print( 'Starting {}...'.format(self.name) )
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.end = timeit.default_timer()
        self.elapsed = self.end - self.start
        if self.verbose:
            print( '{}: elapsed {} sec'.format(self.name, self.elapsed) )


class SaveAs:
    """Use this to ensure an object (e.g. neural net) gets saved even when the program crashes 
    (e.g. if the cluster decides to end my job without warning in the middle of training the network).
    (Update: this doesn't actually work if the cluster kills the job... but may still be useful if it 
    crashes due to an uncaught exception)
    
    See https://preshing.com/20110920/the-python-with-statement-by-example/ for more
    info about context managers"""
    def __init__(self, obj, filename):
        self.obj = obj
        self.filename = filename
    
    def __enter__(self):
        return self
    
    def __exit__(self, typ, val, trac):
        print(typ)
        print(val)
        joblib.dump(self.obj, self.filename)            


def subplots_square(n, force=False):
    """Generates an approximately-square grid of n subplots
    """
    if n > 400 and force==False:
        raise(RuntimeWarning("Too many plots (n={}). To override, pass 'force=True' as second argument.".format(n)))
        
    a = int(np.ceil(np.sqrt(n)))
    b = int(np.round(np.sqrt(n)))  
    
    r = min(a,b)
    c = max(a,b)
    
    fig, ax = plt.subplots(r,c)    
    return fig, ax


def numerical_gradient(f, x, eps=1e-6):   
    """Numerically computes the gradient of f at x using the central difference approximation.
    f:R^d-->R is a single parameter function
    x=[x_1,..,x_d] is a numpy vector
    """
    #TODO: add a **kwargs (or *args) parameter, pass it through to f to allow for additional non-variable parameters
    g = np.empty((f(x).size, x.size))
    for i in range(x.size):
        eps_i = np.zeros(x.size)
        eps_i[i] = eps/2.           
        g[:,i] = (f(x+eps_i) - f(x-eps_i))/eps
    return g.squeeze()