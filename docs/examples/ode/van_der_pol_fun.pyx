# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:35:59 2016

We define a cython function to use with an integrator. To compile this will
typically be something like: 

 cython -I ~/git/odes/ van_der_pol_fun.pyx

 gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
     -I/usr/include/python2.7 -I~/git/odes/scikits/odes/sundials/\
     -o van_der_pol_fun.so van_der_pol_fun.c

@author: benny
"""
from __future__ import division
import numpy as np
cimport numpy as np
#import scikits.odes.sundials.cvode
from scikits.odes.sundials.cvode cimport CV_RhsFunction
#cimport scikits.odes.sundials.cvode
ctypedef np.float_t DTYPE_t

cdef double mu = 1000;
    
# Right-hand side function
cdef class CV_Rhs_van_der_pol_cy(CV_RhsFunction):
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       object userdata = None):

        ydot[0] = y[1]
        ydot[1] = mu*(1.0-y[0]**2)*y[1]-y[0]
        return 0

cpdef np.ndarray[DTYPE_t, ndim=1] van_der_pol_cy(double t, np.ndarray[DTYPE_t, ndim=1] y):
    cdef np.ndarray[DTYPE_t, ndim=1] ydot = np.empty(2, dtype=np.float)
    ydot[0] = y[1]
    ydot[1] = mu*(1.0-y[0]**2)*y[1]-y[0]
    return ydot
