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

class ode
---------

A generic interface class to ordinary differential equation solvers. 
It has the following methods::

    integrator = ode(integrator_name, rhsfn, **options)
    integrator.set_options(options)
    result = integrator.solve(times, init_val_y, user_data)

Alternatively, an init_step, and step method can be used to iterate over a 
solution.

For ode rhsfn is required, this is the right-hand-side equations evaluator
function f, which must satisfy a specific signature.

"""

from __future__ import print_function

#the following will be extended as solvers are loaded
integrator_info = \
"""
Available integrators
---------------------
cvode

Note: 
-----
Consider also the solvers from scipy.integrate, specifically odeint and 
scipy.integrate.ode. 
At the moment of writing, these methods of scipy are based on the original
lsoda/e and vode fortran solvers. cvode is the successor (and improvement) of 
those, with a last release in sundials 2.4.0

""" 

__doc__ += integrator_info

__all__ = ['ode']
__version__ = "$Id$"
__docformat__ = "restructuredtext en"

import re, sys

from numpy import isscalar, array, asarray

class OdeBase(object):
    """ the interface which ODE solvers must implement"""
    
    integrator_classes = []

    def __init__(self, Rfn, **options):
        """ 
        Initialize the ODE Solver and it's default values 

        Input:
            Rfn     - right-hand-side function
            options - additional options for initialization
        """
        raise NotImplementedError('all ODE solvers must implement this')

    def set_options(self, **options):
        """
        Set specific options for the solver.
        
        Calling set_options a second time, normally resets the solver.
        """
        raise NotImplementedError('all ODE solvers must implement this')

    def solve(self, tspan, y0, hook_fn = None):
        """
        Runs the solver.
        
        Input:
            tspan - an list/array of times at which the computed value will be
                    returned. Must contain the start time.
            y0    - list/numpy array of initial values
            hook_fn  - if set, this function is evaluated after each successive 
                       internal) step. Input values: t, x, xdot, userdata. 
                       Output is 0 (success), otherwise computation is stopped 
                      and a return flag = ? is set. Values are stored in (see) t_err, y_err, yp_err
            
        Return values:
            flag   - indicating return status of the solver
            t      - numpy array of times at which the computations were successful
            y      - numpy array of values corresponding to times t (values of y[i, :] ~ t[i])
            t_err  - float or None - if recoverable error occured (for example reached maximum
                     number of allowed iterations), this is the time at which it happened
            y_err  - numpy array of values corresponding to time t_err
        """
        raise NotImplementedError('all ODE solvers must implement this')
    
    def init_step(self, t0, y0):
        """
        Initializes the solver and allocates memory.

        Input:
            t0     - initial time
            y0     - initial condition for y (can be list or numpy array)
        
        Return value:
            t      - initial time at which solver will start.
        """
        raise NotImplementedError('all ODE solvers must implement this')

    def step(self, t, y_retn):
        """
        Method for calling successive next step of the ODE solver to allow
        more precise control over the solver. The 'init_step' method has to
        be called before the 'step' method.
        
        Input:
            t - if t>0.0 then integration is performed until this time
                         and results at this time are returned in y_retn
              - if t<0.0 only one internal step is perfomed towards time abs(t)
                         and results after this one time step are returned
            y_retn - numpy vector (ndim = 1) in which the computed
                     value will be stored  (needs to be preallocated)
        Return values:
            flag  - status of the computation (successful or error occured)
            t_out - time, where the solver stopped (when no error occured, t_out == t)

        """
        raise NotImplementedError('all ODE solvers must implement this')

#------------------------------------------------------------------------------
# User interface
#------------------------------------------------------------------------------

class ode(object):
    """\
A generic interface class to differential equation solvers.

See also
--------
scipy.integrate.odeint : an ODE integrator with a simpler interface based on lsoda from ODEPACK
scipy.integrate.ode : class around vode ODE integrator

Examples
--------
ODE arise in many applications of dynamical systems, as well as in 
discritisations of PDE (eg moving mesh combined with method of 
lines). 
As an easy example, consider the simple oscillator, 

>>> from __future__ import print_function
>>> from numpy import cos, sin, sqrt
>>> k = 4.0
>>> m = 1.0
>>> initx = [1, 0.1]
>>> def rhseqn(t, x, result):
    # we create rhs equations for the problem
      result[0] = - k/m*x[0]
      result[1] = x[1]
    
>>> from scikits.odes import ode
>>> solver = ode('cvode', rhseqn)
>>> result = solver.solve([0., 1., 2.], initx)
>>> print('\n   t        Solution          Exact')
>>> print('------------------------------------')
>>> for t, u in zip(result[1], result[2]):
        print('%4.2f %15.6g %15.6g' % (t, u[0], initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m)))

"""
    __doc__ += integrator_info
    LOADED = False

    def __init__(self, integrator_name, eqsrhs, **options):
        """
        Initialize the ODE Solver and it's default values


        Parameters
        ----------
        integrator_name : name of the integrator solver to use. Currently you 
            can choose cvode.
        eqsrhs : right-hand-side function
            right-hand-side of a first order ode. The signature of this
            function depends on the solver used, see the solver documentation 
            for details.
            Generally however, you can assume the following signature to work:
                        eqsrhs(x, y, return_rhs)
            with 
            x       : independent variable, eg the time, float
            y       : array of n unknowns in x
            return_rhs: array that must be updated with the value of the 
                      right-hand-side, so f(t,y).  The dimension is equal to 
                      dim(y)
            return value: An integer, 0 for success. It is not guaranteed that 
                      a solver takes this status into account
        
            Some solvers will allow userdata to be passed to eqsrhs, or optional
            formats that are more performant.
        options :  additional options of the solver, see set_options method of
            the solver for details.
        """

        integrator = find_ode_integrator(integrator_name)
        if integrator is None:
            raise ValueError('No integrator name match with %s or is not available.'\
                  %(repr(integrator_name)))
        else:
            self._integrator = integrator(eqsrhs, **options)

    def set_options(self, **options):
        """
        Set specific options for the solver.
        See the solver documentation for details.
        
        Calling set_options a second time, normally resets the solver.
        """
        return self._integrator.set_options(**options)

    def solve(self, tspan, y0, hook_fn = None):
        """
        Runs the solver.
        
        Input:
            tspan - an list/array of times at which the computed value will be
                    returned. Must contain the start time.
            y0    - list/numpy array of initial values
            hook_fn  - if set, this function is evaluated after each successive 
                       internal) step. Input values: t, x, xdot, userdata. 
                       Output is 0 (success), otherwise computation is stopped 
                      and a return flag = ? is set. Values are stored in (see) t_err, y_err, yp_err
            
        Return values:
            flag   - indicating return status of the solver
            t      - numpy array of times at which the computations were successful
            y      - numpy array of values corresponding to times t (values of y[i, :] ~ t[i])
            t_err  - float or None - if recoverable error occured (for example reached maximum
                     number of allowed iterations), this is the time at which it happened
            y_err  - numpy array of values corresponding to time t_err
        """
        return self._integrator.solve(tspan, y0, hook_fn)

    def init_step(self, t0, y0):
        """
        Initializes the solver and allocates memory.

        Input:
            t0     - initial time
            y0     - initial condition for y (can be list or numpy array)
        
        Return value:
            t      - initial time at which solver will start.
        """
        return self._integrator.init_step(t0, y0)

    def step(self, t, y_retn):
        """
        Method for calling successive next step of the ODE solver to allow
        more precise control over the solver. The 'init_step' method has to
        be called before the 'step' method.
        
        Input:
            t - if t>0.0 then integration is performed until this time
                         and results at this time are returned in y_retn
              - if t<0.0 only one internal step is perfomed towards time abs(t)
                         and results after this one time step are returned
            y_retn - numpy vector (ndim = 1) in which the computed
                     value will be stored  (needs to be preallocated)
        Return values:
            flag  - status of the computation (successful or error occured)
            t_out - time, where the solver stopped (when no error occured, t_out == t)

        """
        return self._integrator.step(t, y_retnn)

#------------------------------------------------------------------------------
# ODE integrators
#------------------------------------------------------------------------------


integrator_info_cvode = """
CVODE solver from the SUNDIALS package. See info in 
scikits.odes.sundials.cvode.CVODE class
"""
__doc__ += integrator_info_cvode
integrator_info += integrator_info_cvode

def find_ode_integrator(name):
    if not ode.LOADED:
        ## cvode
        try:
            from sundials import cvode
            OdeBase.integrator_classes.append(cvode.CVODE)
        except ValueError as msg:
            print('Could not load CVODE solver', msg)
        except ImportError:
            print(sys.exc_info()[1])

        ode.LOADED = True

    for cl in OdeBase.integrator_classes:
        if re.match(name, cl.__name__, re.I):
            return cl
        elif hasattr(cl, name) and re.match(name, cl.name, re.I):
            return cl
    raise ValueError('Integrator name %s does not exsist' % name)
