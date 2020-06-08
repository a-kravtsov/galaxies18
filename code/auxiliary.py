# Auxiliary routines such as integration
#   
import numpy as np


def Rmm(a, b, func, m, **kwargs):
    """
    Auxiliary function computing tableau entries for Romberg integration
    using recursive relation, but implemented non-recursively
    
    Parameters
    -----------------
    func - python function object
            function to integrate
    a, b - floats
            integration interval            
    m    - integer
            iteration level; accuracy order will be equal to 2(m+1)
            in this implementation there is no need for k on input
            
    kwargs - python dictionary 
            array of keyword arguments to be passed to the integrated function
            
    Returns
    ---------
    
    I(m)   - float
              estimate of the integral using scheme of order 2*m+2
    I(m-1) - float
              estimate of the integral using scheme of order 2*m
    """
    assert(m >= 0)
    
    ba = b - a;
    hk = ba / 2**(np.arange(m+1)) # vector of step sizes

    Rkm = np.zeros((m+1,m+1)) 

    Rkm[0,0] = 0.5 * ba * (func(a, **kwargs) + func(b, **kwargs))
        
    for k in range(1,m+1):
        # first compute R[k,0]
        trapzd_sum = 0.
        for i in range(1, 2**(k-1)+1):
            trapzd_sum += func(a + (2*i-1)*hk[k], **kwargs)
            
        # we can reuse Rkm[k-1,0] but we need to divide it by 2 to account for step decrease 
        Rkm[k,0] = Rkm[k-1,0] * 0.5 + hk[k] * trapzd_sum
        
        # then fill the tableau up to R[k,k]
        for md in range(1,k+1):
            fact = 4.**md
            Rkm[k,md] = (fact * Rkm[k,md-1] - Rkm[k-1,md-1])/(fact - 1)

          
    return Rkm[m,m], Rkm[m,m-1] # return the desired approximation and best one of previous order 

def romberg(func, a, b, rtol = 1.e-4, mmax = 8, verbose = False, **kwargs):
    """
    Romberg integration scheme to evaluate
            int_a^b func(x)dx 
    using recursive relation to produce higher and higher order approximations
    
    Code iterates from m=0, increasing m by 1 on each iteration.
    Each iteration computes the integral using scheme of 2(m+2) order of accuracy 
    Routine checks the difference between approximations of successive orders
    to estimate error and stops when a desired relative accuracy 
    tolerance is reached.
    
    - Andrey Kravtsov, 2017

    Parameters
    --------------------------------
    
    func - python function object
            function to integrate
    a, b - floats
            integration interval
    rtol - float 
            fractional tolerance of the integral estimate
    mmax - integer
            maximum number of iterations to do 
    verbose - logical
            if True print intermediate info for each iteration
    kwargs - python dictionary
             a list of parameters with their keywords to pass to func
               
    Returns
    ---------------------------------
    I    - float
           estimate of the integral for input f, [a,b] and rtol
    err  - float 
           estimated fractional error of the estimated integral

    """
    assert(a < b)
    
    for m in range(1, mmax):
        Rmk_m, Rmk_m1 = Rmm(a, b, func, m, **kwargs)
            
        if Rmk_m == 0:
            Rmk_m = 1.e-300 # guard against division by 0 
            
        etol = 1.2e-16 + rtol*np.abs(Rmk_m)
        err = np.abs(Rmk_m-Rmk_m1)

        if verbose: 
            print("m = %d, integral = %.6e, prev. order = %.6e, frac. error estimate = %.3e"%(m, Rmk_m, Rmk_m1, err/Rmk_m))

        if (m>0) and (np.abs(err) <= etol):
            return Rmk_m, err/Rmk_m
        
    print("!!! Romberg warning: !!!")
    print("!!! maximum of mmax=%d iterations reached, abs(err)=%.3e, > required error rtol = %.3e"%(mmax, np.abs(err/Rmk_m), rtol))
    return Rmk_m, err/Rmk_m
    
from scipy.misc import comb
import numpy as np
import sys 

def choose(n, k):
    """
    A fast way to calculate binomial coefficient by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

def compute_wi(xi):
    """
    compute general coefficients for barycentric interpolation for arbitrary point
    spacing
    """
    Np = np.size(xi)
    wi = np.ones(Np)
    for i in range(Np):
        for j in range(Np):
            if i != j:
                wi[i] *= (xi[i] - xi[j])
                
    return 1./wi

def compute_wi_equidistant(Np):
    """
    barycentric weights for equidistant points on any interval
    """
    wi = np.zeros(Np)
    for i in range(Np):
        wi[i] = (-1.)**i*choose(Np,i)
    return wi

def compute_wi_chebyshev2(Np):
    """
    barycentric weights for Chebyshev points of the 2nd kind on any interval
    """
    wi = np.zeros(Np)
    for i in range(1,Np-1):
        wi[i] = (-1.)**i
    wi[0], wi[Np-1] = 0.5, 0.5*(-1)**(Np-1); # edges are different 
    return wi

def bar_polyint(xi, fi, wi, f, x):
    """
    barycentric interpolation
    """
    Np = np.size(xi); Nt = np.size(x)
    ip = np.arange(Np)
    sumn, sumd = np.zeros_like(np.array(x)), np.zeros_like(np.array(x))
    exact = np.zeros(np.size(np.array(x)), dtype=bool)
    for i in range(Np):
        dx = x - xi[i]
        dxsign = np.sign(dx); dxabs = np.abs(dx)
        inon0 = (dxabs>0)
        sumn[inon0] += wi[i]*fi[i]*dxsign[inon0]/dxabs[inon0] 
        sumd[inon0] += wi[i]*dxsign[inon0] / dxabs[inon0] 
        if Nt > 1: 
            exact[dxabs==0] = True # flag points right on the nodes
        else:
            exact[0] = True
            
    px = sumn / sumd
    
    if Nt > 1:
        px[exact] = f(x[exact])
    else: # if a single point right on the node
        if exact == True: 
            px = f(x)
            
    return px
    
