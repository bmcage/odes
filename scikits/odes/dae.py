# Authors: B. Malengier based on ode.py
"""
First-order DAE solver

User-friendly interface to various numerical integrators for solving an
algebraic system of first order ODEs with prescribed initial conditions:

         d y(t)
    A * ---------  = f(t,y(t)),
            d t

    y(t=0)[i] = y0[i],
    
      d y(t=0)
    ---------- [i]  = yprime0[i],
        d t

where::

    i = 0, ..., len(y0) - 1
    A is a (possibly singular) matrix of size i x i
    f(t,y) is a vector of size i
    
or more generally, equations of the form 

    G(t,y,y') = 0

class dae
---------

A generic interface class to differential algebraic equations. 
It has the following methods::

    integrator = dae(integrator_name, resfn, **options)
    integrator.set_options(options)
    result = integrator.solve(times, init_val_y, init_val_yp, user_data)

Alternatively, an init_step, and step method can be used to iterate over a 
solution.

For dae resfn is required, this is the residual equations evaluator
function, which must satisfy a specific signature.
"""

from __future__ import print_function

integrator_info = \
"""
Available integrators
---------------------
ida
ddaspk
lsodi 

Possibilities for the future:

ddaskr  
~~~~~~
Not included, starting hints:
                 http://osdir.com/ml/python.f2py.user/2005-07/msg00014.html

Modified Extended Backward Differentiation Formulae (MEBDF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Not included. Fortran codes: http://www.ma.ic.ac.uk/~jcash/IVP_software/readme.html
"""
__doc__ += integrator_info

__all__ = ['dae']
__version__ = "$Id$"
__docformat__ = "restructuredtext en"

from numpy import asarray, array, zeros, sin, int32, isscalar, empty, alen
from copy import copy
import re, sys

class DaeBase(object):
    """ the interface which DAE solvers must implement"""
    
    integrator_classes = []

    def __init__(self, Rfn, **options):
        """
        Initialize the DAE Solver and it's default values

        Input:
            Rfn     - residual function
            options - additional options for initialization
        """
        raise NotImplementedError('all DAE solvers must implement this')

    def set_options(self, **options):
        """
        Set specific options for the solver.
        
        Calling set_options a second time, normally resets the solver.
        """
        raise NotImplementedError('all DAE solvers must implement this')

    def solve(self, tspan, y0,  yp0, hook_fn = None):
        """
        Runs the solver.
        
        Input:
            tspan - an list/array of times at which the computed value will be
                    returned. Must contain the start time.
            y0    - list/numpy array of initial values
            yp0   - list/numpy array of initial values of derivatives
            hook_fn  - if set, this function is evaluated after each successive 
                       internal) step. Input values: t, x, xdot, userdata. 
                       Output is 0 (success), otherwise computation is stopped 
                      and a return flag = ? is set. Values are stored in (see) t_err, y_err, yp_err
            
        Return values:
            flag   - indicating return status of the solver
            t      - numpy array of times at which the computations were successful
            y      - numpy array of values corresponding to times t (values of y[i, :] ~ t[i])
            yp     - numpy array of derivatives corresponding to times t (values of yp[i, :] ~ t[i])
            t_err  - float or None - if recoverable error occured (for example reached maximum
                     number of allowed iterations), this is the time at which it happened
            y_err  - numpy array of values corresponding to time t_err
            yp_err - numpy array of derivatives corresponding to time t_err
            
        Note:
            If 'calc_initcond' option set, then solver returns instead of user 
            supplied y0, yp0 values as the starting values the values calculated 
            by the solver (i.e. consistent initial
            conditions. The starting time is then also the precomputed time.
        """
        raise NotImplementedError('all DAE solvers must implement this')

    def init_step(self, t0, y0, yp0, y_ic0_retn = None, yp_ic0_retn = None):
        """
        Initializes the solver and allocates memory.

        Input:
            t0     - initial time
            y0     - initial condition for y (can be list or numpy array)
            yp0    - initial condition for yp (can be list or numpy array)
            y_ic0  - (optional) returns the calculated consistent initial 
                     condition for y
                     It MUST be a numpy array.
            yp_ic0 - (optional) returns the calculated consistent initial
                     condition for y derivated. It MUST be a numpy array.
        
        Return Value:
            t      - time of solver at end of init_step, from which solver will
                     continue
        """
        raise NotImplementedError('all DAE solvers must implement this')

    def step(self, t, y_retn, yp_retn = None):
        """
        Method for calling successive next step of the IDA solver to allow
        more precise control over the IDA solver. The 'init_step' method has to
        be called before the 'step' method.
        
        Input:
            t - if t>0.0 then integration is performed until this time
                         and results at this time are returned in y_retn
              - if t<0.0 only one internal step is perfomed towards time abs(t)
                         and results after this one time step are returned
            y_retn - numpy vector (ndim = 1) in which the computed
                     value will be stored  (needs to be preallocated)
            yp_retn - numpy vector (ndim = 1) or None. If not None, will be
                      filled (needs to be preallocated)
                      with derivatives of y at time t.
        Return values:
            flag  - status of the computation (successful or error occured)
            t_out - time, where the solver stopped (when no error occured, t_out == t)
        """
        raise NotImplementedError('all DAE solvers must implement this')

#------------------------------------------------------------------------------
# User interface
#------------------------------------------------------------------------------

class dae(object):
    """\
A generic interface class to differential algebraic equation solvers.

See also
--------
odeint : an ODE integrator with a simpler interface based on lsoda from ODEPACK
ode : class around vode ODE integrator

Examples
--------
DAE arise in many applications of dynamical systems, as well as in 
discritisations of PDE (eg moving mesh combined with method of 
lines). 
As an easy example, consider the simple oscillator, which we write as 
G(y,y',t) = 0 instead of the normal ode, and solve as a DAE.

>>> from __future__ import print_function
>>> from numpy import cos, sin, sqrt
>>> k = 4.0
>>> m = 1.0
>>> initx = [1, 0.1]
>>> initxp = [initx[1], -k/m*initx[0]]
>>> def reseqn(t, x, xdot, result):
    # we create residual equations for the problem
      result[0] = m*xdot[1] + k*x[0]
      result[1] = xdot[0] - x[1]
    
>>> from scikits.odes import dae
>>> solver = dae('ida', reseqn)
>>> result = solver.solve([0., 1., 2.], initx, initxp)
>>> print('\n   t        Solution          Exact')
>>> print('------------------------------------')
>>> for t, u in zip(result[1], result[2]):
        print('%4.2f %15.6g %15.6g' % (t, u[0], initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m)))

"""
    __doc__ += integrator_info
    LOADED = False

    def __init__(self, integrator_name, eqsres, **options):
        """
        Initialize the DAE Solver and it's default values

        Define equation res = G(t,y,y') which can eg be G = f(y,t) - A y' when 
        solving A y' = f(y,t), 
        and where (optional) jac is the jacobian matrix of the nonlinear system
        see fortran source code), so d res/dy + scaling * d res/dy' or d res/dy
        depending on the backend

        Parameters
        ----------
        integrator_name : name of the integrator solver to use. Currently you 
            can choose ida, ddaspk and lsodi, with ida the most recent solver.
        eqsres : residual function
            Residual of the DAE. The signature of this function depends on the
            solver used, see the solver documentation for details.
            Generally however, you can assume the following signature to work:
                        eqsres(x, y, yprime, return_residual)
            with 
            x       : independent variable, eg the time, float
            y       : array of n unknowns in x
            yprime  : dy/dx array of n unknowns in x, dimension = dim(y)
            return_residual: array that must be updated with the value of the 
                      residuals, so G(t,y,y').  The dimension is equal to dim(y)
            return value: An integer, 0 for success. It is not guaranteed that 
                      a solver takes this status into account
        
            Some solvers will allow userdata to be passed to eqsres, or optional
            formats that are more performant.
        options :  additional options of the solver, see set_options method of
            the solver for details.
        """

        integrator = find_dae_integrator(integrator_name)
        if integrator is None:
            raise ValueError('No integrator name match with %s or is not available.'\
                  %(repr(integrator_name)))
        else:
            self._integrator = integrator(eqsres, **options)

    def set_options(self, **options):
        """
        Set specific options for the solver.
        See the solver documentation for details.
        
        Calling set_options a second time, normally resets the solver.
        """
        return self._integrator.set_options(**options)

    def solve(self, tspan, y0,  yp0, hook_fn = None):
        """
        Runs the solver.
        
        Input:
            tspan - an list/array of times at which the computed value will be
                    returned. Must contain the start time.
            y0    - list/numpy array of initial values
            yp0   - list/numpy array of initial values of derivatives
            hook_fn  - if set, this function is evaluated after each successive 
                       internal) step. Input values: t, x, xdot, userdata. 
                       Output is 0 (success), otherwise computation is stopped 
                      and a return flag = ? is set. Values are stored in (see) t_err, y_err, yp_err
            
        Return values:
            flag   - indicating return status of the solver
            t      - numpy array of times at which the computations were successful
            y      - numpy array of values corresponding to times t (values of y[i, :] ~ t[i])
            yp     - numpy array of derivatives corresponding to times t (values of yp[i, :] ~ t[i])
            t_err  - float or None - if recoverable error occured (for example reached maximum
                     number of allowed iterations), this is the time at which it happened
            y_err  - numpy array of values corresponding to time t_err
            yp_err - numpy array of derivatives corresponding to time t_err
            
        Note:
            If 'calc_initcond' option set, then solver returns instead of user 
            supplied y0, yp0 values as the starting values the values calculated 
            by the solver (i.e. consistent initial
            conditions. The starting time is then also the precomputed time.
        """
        return self._integrator.solve(tspan, y0,  yp0, hook_fn)

    def init_step(self, t0, y0, yp0, y_ic0_retn = None, yp_ic0_retn = None):
        """
        Initializes the solver and allocates memory. It is not needed to 
        call this method if solve is used to compute the solution. In the case
        step is used, init_step must be called first.

        Input:
            t0     - initial time
            y0     - initial condition for y (can be list or numpy array)
            yp0    - initial condition for yp (can be list or numpy array)
            y_ic0  - (optional) returns the calculated consistent initial condition for y
                     It MUST be a numpy array.
            yp_ic0 - (optional) returns the calculated consistent initial
                     condition for y derivated. It MUST be a numpy array.
        """
        return self._integrator.init_step(t0, y0, yp0, y_ic0_retn, yp_ic0_retn)

    def step(self, t, y_retn, yp_retn = None):
        """
        Method for calling successive next step of the solver to allow
        more precise control over the solver. The 'init_step' method has to
        be called before the 'step' method.
        
        Input:
            t - if t>0.0 then integration is performed until this time
                         and results at this time are returned in y_retn
              - if t<0.0 only one internal step is perfomed towards time abs(t)
                         and results after this one time step are returned
            y_retn - numpy vector (ndim = 1) in which the computed
                     value will be stored  (needs to be preallocated)
            yp_retn - numpy vector (ndim = 1) or None. If not None, will be
                      filled (needs to be preallocated)
                      with derivatives of y at time t.
        Return values:
            flag  - status of the computation (successful or error occured)
            t_out - time, where the solver stopped (when no error occured, t_out == t)
        """
        return self._integrator.step(t, y_retn, yp_retn)

#------------------------------------------------------------------------------
# DAE integrators
#------------------------------------------------------------------------------

integrator_info_ida = """
            IDA solver from the SUNDIALS package. See info in 
            scikits.odes.sundials.ida.IDA class
            """
__doc__ += integrator_info_ida
integrator_info += integrator_info_ida

def find_dae_integrator(name):
    if not dae.LOADED:
        ## ida
        try:
            from .sundials import ida
            DaeBase.integrator_classes.append(ida.IDA)
        except ValueError as msg:
            print('Could not load IDA solver', msg)
        except ImportError:
            print(sys.exc_info()[1])

        ## ddaspk
        try:
            from .ddaspkint import ddaspk
        except ImportError:
            print(sys.exc_info()[1])

        ## lsodi
        try:
            from .lsodiint import lsodi
        except ImportError:
            print(sys.exc_info()[1])

        dae.LOADED = True
        
    for cl in DaeBase.integrator_classes:
        if re.match(name, cl.__name__, re.I):
            return cl
        elif hasattr(cl, name) and re.match(name, cl.name, re.I):
            return cl
    raise ValueError('Integrator name %s does not exsist' % name)
