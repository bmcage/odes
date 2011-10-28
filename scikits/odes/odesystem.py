# Authors: B. Malengier based on ode.py
"""
First-order ODE solver

User-friendly interface to various numerical integrators for solving a
system of first order ODEs with prescribed initial conditions:

     d y(t)
    ---------  = f(t,y(t)),
      d t

    y(t=0)[i] = y0[i],

where::

    i = 0, ..., len(y0) - 1
    f(t,y) is a vector of size i

class ODEsystem
---------------

A generic interface class to ordinary differential equation solvers. 
It has the following methods::

    integrator = ODEsystem(rhs,jac=None)
    integrator = integrator.set_integrator(name, **params)
    integrator = integrator.set_initial_value(y0, t=0.0)
    y1 = integrator.integrate(t1,step=0,relax=0)
    flag = integrator.successful()
    if not flag:
        print 'return message', integrator.successmsg

rhs and jac need to have the signature as required by the integrator name. If
you need to pass extra arguments to jac, use eg a python class method : 
    problem = Myproblem()
    integrator = ODEsystem(problem.res, problem.jac)
Allowing the extra parameters to be kept in the Myproblem class
"""

from __future__ import print_function

#the following will be extended as solvers are loaded
integrator_info = \
"""
Available integrators
---------------------
""" 

__doc__ += integrator_info

__all__ = []
__version__ = "$Id: odes_cvode bmalengier $"
__docformat__ = "restructuredtext en"

import re

from scipy.integrate import ode
from scipy.integrate._ode import IntegratorBase
from numpy import isscalar, array, asarray

##try:
##    from .odes_cvode import odesCVODE, integrator_info_cvode
##    IntegratorBase.integrator_classes.append(odesCVODE)
##    __doc__ += integrator_info_cvode
##    integrator_info += integrator_info_cvode
##except:
##    print('Could not load odesCVODE')

class ODEsystem(ode):
    __doc__ = ode.__doc__ + integrator_info

    def set_integrator(self, name, **integrator_params):
        """
        Set integrator by name.

        Parameters
        ----------
        name : str
            Name of the integrator
        integrator_params
            Additional parameters for the integrator.
        """
        integrator = find_odes_integrator(name)
        if integrator is None:
            print('No integrator name match with %s or is not available.'\
                  %(repr(name)))
        else:
            self._integrator = integrator(**integrator_params)
            if not len(self.y):
                self.t = 0.0
                self.y = array([0.0], self._integrator.scalar)
        return self
    
    def set_initial_value(self, y, t=0.0):
        """Set initial conditions y(t) = y."""
        if isscalar(y):
            y = [y]
        y = asarray(y, self._integrator.scalar)
        #if method needs to initialize init val itself, do that
        print('set_initial_value called')
        print(hasattr(self._integrator, 'set_init_val'))
        if hasattr(self._integrator, 'set_init_val'):
            self._integrator.set_init_val(y, t, 
                                        self.f, self.jac)
            
        ode.set_initial_value(self, y, t)
        return self
    
    def set_f_params(self, *args):
        """You cannot pass f parameters. Use a class function instead."""
        raise NotImplementedError

    def set_jac_params(self,*args):
        """You cannot pass f parameters. Use a class function instead."""
        raise NotImplementedError

def find_odes_integrator(name):
    for cl in IntegratorBase.integrator_classes:
        try:
            if re.match(name, cl.__name__, re.I) or re.match(name, cl.name, re.I):
                return cl
        except AttributeError:
            #no cl.name attribute
            pass
    return

    for cl in IntegratorBase.integrator_classes:
        if re.match(name,cl.__name__,re.I):
            return cl
    return
