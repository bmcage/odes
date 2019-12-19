# -*- coding: utf-8 -*-
# Authors: B. Malengier based on ode.py
r"""
First-order DAE solver
======================

User-friendly interface to various numerical integrators for solving an
algebraic system of first order ODEs with prescribed initial conditions:

.. math::
    A \frac{dy(t)}{dt} = f(t,y(t)),

    y(t=0)[i] = y0[i],

    \frac{d y(t=0)}{dt}[i]  = yprime0[i],

where :math:`i = 0, ..., len(y0) - 1`; :math:`A` is a (possibly singular) matrix
of size :math:`i × i`; and :math:`f(t,y)` is a vector of size :math:`i` or more generally, equations of the form

.. math::
    G(t,y,y') = 0

"""

from __future__ import print_function

__all__ = ['dae']
__version__ = "$Id$"
__docformat__ = "restructuredtext en"

from numpy import asarray, array, zeros, sin, int32, isscalar, empty, alen
from copy import copy
import re, sys

class DaeBase(object):
    """
    The interface which DAE solvers must implement.

    Parameters
    ----------
    Rfn :
        residual function
    options : mapping
        Additional options for initialization, solver dependent
    """

    integrator_classes = []

    def __init__(self, Rfn, **options):
        raise NotImplementedError('all DAE solvers must implement this')

    def set_options(self, **options):
        """
        Set specific options for the solver.

        Calling set_options a second time, normally resets the solver.
        """
        raise NotImplementedError('all DAE solvers must implement this')

    def solve(self, tspan, y0,  yp0):
        """
        Runs the solver.

        Parameters
        ----------
            tspan : list/array
                A list of times at which the computed value will be returned. Must contain the start time as first entry.
            y0 : list/array
                List of initial values
            yp0 : list/array
                List of initial values of derivatives

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnumXXX)
            ``values``  Named tuple with entries array t and array y and array ydot. y will correspond to y_retn value and ydot to yp_retn!
            ``errors``  Named tuple with entries t and y and ydot of error
            ``roots``   Named tuple with entries array t and array y and array ydot
            ``tstop``   Named tuple with entries array t and array y and array ydot
            ``message`` String with message in case of an error
            =========== ==========================================

        old_api is True : tuple
            tuple with the following elements in order

            ========== ==========================================
            Field      Meaning
            ========== ==========================================
            ``flag``   indicating return status of the solver
            ``t``      numpy array of times at which the computations were successful
            ``y``      numpy array of values corresponding to times t (values of y[i, :] ~ t[i])
            ``yp``     numpy array of derivatives corresponding to times t (values of yp[i, :] ~ t[i])
            ``t_err``  float or None - if recoverable error occured (for example reached maximum number of allowed iterations), this is the time at which it happened
            ``y_err``  numpy array of values corresponding to time t_err
            ``yp_err`` numpy array of derivatives corresponding to time t_err
            ========== ==========================================

        """
        raise NotImplementedError('all DAE solvers must implement this')

    def init_step(self, t0, y0, yp0, y_ic0_retn = None, yp_ic0_retn = None):
        """
        Initializes the solver and allocates memory.

        Parameters
        ----------
        t0 : number
            initial time
        y0 : list/array
            initial condition for y
        yp0 :  list/array
            initial condition for yp
        y_ic0 : numpy array
            (optional) returns the calculated consistent initial condition for y
        yp_ic0 :  numpy array
            (optional) returns the calculated consistent initial condition for y derivated.

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnumXXX)
            ``values``  Named tuple with entries t and y and ydot. y will correspond to y_retn value and ydot to yp_retn!
            ``errors``  Named tuple with entries t and y and ydot
            ``roots``   Named tuple with entries t and y and ydot
            ``tstop``   Named tuple with entries t and y and ydot
            ``message`` String with message in case of an error
            =========== ==========================================

        old_api is True : tuple
            tuple with the following elements in order

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    status of the computation (successful or error occured)
            ``t_out``   time, where the solver stopped (when no error occured, t_out == t)
            =========== ==========================================

        """
        raise NotImplementedError('all DAE solvers must implement this')

    def step(self, t, y_retn=None, yp_retn=None):
        """
        Method for calling successive next step of the IDA solver to allow
        more precise control over the IDA solver. The 'init_step' method has to
        be called before the 'step' method.

        A step is done towards time t, and output at t returned. This time can be higher or lower than the previous time. If option 'one_step_compute'==True, and the solver supports it, only one internal solver step is done in the direction of t starting at the current step.

        If old_api=True, the old behavior is used: if t>0.0 then integration is
        performed until this time and results at this time are returned in
        y_retn; else if if t<0.0 only one internal step is perfomed towards time
        abs(t) and results after this one time step are returned.

        Parameters
        ----------
        t : number
        y_retn : numpy array (ndim = 1) or None.
            (Needs to be preallocated) If not None, will be filled with y at time t. If None y_retn is not used.
        yp_retn : numpy array (ndim = 1) or None.
            (Needs to be preallocated) If not None, will be filled with derivatives of y at time t. If None yp_retn is not used.

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnumXXX)
            ``values``  Named tuple with entries t and y and ydot. y will correspond to y_retn value and ydot to yp_retn!
            ``errors``  Named tuple with entries t and y and ydot
            ``roots``   Named tuple with entries t and y and ydot
            ``tstop``   Named tuple with entries t and y and ydot
            ``message`` String with message in case of an error
            =========== ==========================================

        old_api is True : tuple
            tuple with the following elements in order

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    status of the computation (successful or error occured)
            ``t_out``   time, where the solver stopped (when no error occured, t_out == t)
            =========== ==========================================

        """
        raise NotImplementedError('all DAE solvers must implement this')

#------------------------------------------------------------------------------
# User interface
#------------------------------------------------------------------------------

class dae(object):
    """
    A generic interface class to differential algebraic equations.

    Define equation res = G(t,y,y') which can eg be G = f(y,t) - A y' when solving A y' = f(y,t), and where (optional) jac is the jacobian matrix of the nonlinear system see fortran source code), so d res/dy + scaling * d res/dy' or d res/dy depending on the backend.

    Parameters
    ----------
    integrator_name : ``'ida'``, ``'ddaspk'`` or ``'lsodi'``
        The integrator solver to use.

    eqsres : residual function
        Residual of the DAE. The signature of this function depends on the
        solver used, see the solver documentation for details.
        Generally however, you can assume the following signature to work:

            ``eqsres(x, y, yprime, return_residual)``

        with
        x       : independent variable, eg the time, float
        y       : array of n unknowns in x
        yprime  : dy/dx array of n unknowns in x, dimension = dim(y)
        return_residual: array that must be updated with the value of the residuals, so G(t,y,y').  The dimension is equal to dim(y)
        return value: integer, 0 for success. It is not guaranteed that a solver takes this status into account

        Some solvers will allow userdata to be passed to eqsres, or optional
        formats that are more performant.

    options : mapping
        Additional options for initialization, solver dependent
        See set_options method of the `integrator_name` you selected for
        details.


    See Also
    --------
    odeint : an ODE integrator with a simpler interface based on lsoda from ODEPACK
    ode : class around vode ODE integrator


    Notes
    -----
    Possible future solvers

    ddaskr: Not included, starting hints: http://osdir.com/ml/python.f2py.user/2005-07/msg00014.html
    Modified Extended Backward Differentiation Formulae (MEBDF): Not included. Fortran codes: http://www.ma.ic.ac.uk/~jcash/IVP_software/readme.html

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
        ... # we create residual equations for the problem
        ... result[0] = m*xdot[1] + k*x[0]
        ... result[1] = xdot[0] - x[1]
    >>> from scikits.odes import dae
    >>> solver = dae('ida', reseqn)
    >>> result = solver.solve([0., 1., 2.], initx, initxp)
    """

    LOADED = False

    def __init__(self, integrator_name, eqsres, **options):
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

    def solve(self, tspan, y0,  yp0):
        """
        Runs the solver.

        Parameters
        ----------
        tspan : list/array
            A list of times at which the computed value will be returned. Must contain the start time as first entry.
        y0 : list/array
            list array of initial values
        yp0 : list/array
            list array of initial values of derivatives

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnumXXX)
            ``values``  Named tuple with entries array t and array y and array ydot. y will correspond to y_retn value and ydot to yp_retn!
            ``errors``  Named tuple with entries t and y and ydot of error
            ``roots``   Named tuple with entries array t and array y and array ydot
            ``tstop``   Named tuple with entries array t and array y and array ydot
            ``message`` String with message in case of an error
            =========== ==========================================

        old_api is True : tuple
            tuple with the following elements in order

            ========== ==========================================
            Field      Meaning
            ========== ==========================================
            ``flag``   indicating return status of the solver
            ``t``      numpy array of times at which the computations were successful
            ``y``      numpy array of values corresponding to times t (values of y[i, :] ~ t[i])
            ``yp``     numpy array of derivatives corresponding to times t (values of yp[i, :] ~ t[i])
            ``t_err``  float or None - if recoverable error occured (for example reached maximum number of allowed iterations), this is the time at which it happened
            ``y_err``  numpy array of values corresponding to time t_err
            ``yp_err`` numpy array of derivatives corresponding to time t_err
            ========== ==========================================

        """
        return self._integrator.solve(tspan, y0,  yp0)

    def init_step(self, t0, y0, yp0, y_ic0_retn = None, yp_ic0_retn = None):
        """
        Initializes the solver and allocates memory. It is not needed to
        call this method if solve is used to compute the solution. In the case
        step is used, init_step must be called first.

        Parameters
        ----------
        t0 : number
            initial time
        y0 : list/array
            initial condition for y
        yp0 :  list/array
            initial condition for yp
        y_ic0 : numpy array
            (optional) returns the calculated consistent initial condition for y
        yp_ic0 :  numpy array
            (optional) returns the calculated consistent initial condition for y derivated.

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnumXXX)
            ``values``  Named tuple with entries t and y and ydot. y will correspond to y_retn value and ydot to yp_retn!
            ``errors``  Named tuple with entries t and y and ydot
            ``roots``   Named tuple with entries t and y and ydot
            ``tstop``   Named tuple with entries t and y and ydot
            ``message`` String with message in case of an error
            =========== ==========================================

        old_api is True : tuple
            tuple with the following elements in order

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    status of the computation (successful or error occured)
            ``t_out``   time, where the solver stopped (when no error occured, t_out == t)
            =========== ==========================================

        """
        return self._integrator.init_step(t0, y0, yp0, y_ic0_retn, yp_ic0_retn)

    def step(self, t, y_retn=None, yp_retn=None):
        """
        Method for calling successive next step of the IDA solver to allow
        more precise control over the IDA solver. The 'init_step' method has to
        be called before the 'step' method.

        A step is done towards time t, and output at t returned. This time can be higher or lower than the previous time. If option 'one_step_compute'==True, and the solver supports it, only one internal solver step is done in the direction of t starting at the current step.

        If old_api=True, the old behavior is used: if t>0.0 then integration is
        performed until this time and results at this time are returned in
        y_retn; else if if t<0.0 only one internal step is perfomed towards time
        abs(t) and results after this one time step are returned.

        Parameters
        ----------
        t : number
        y_retn : numpy array (ndim = 1) or None.
            (Needs to be preallocated) If not None, will be filled with y at time t. If None y_retn is not used.
        yp_retn : numpy array (ndim = 1) or None.
            (Needs to be preallocated) If not None, will be filled with derivatives of y at time t. If None yp_retn is not used.

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnumXXX)
            ``values``  Named tuple with entries t and y and ydot. y will correspond to y_retn value and ydot to yp_retn!
            ``errors``  Named tuple with entries t and y and ydot
            ``roots``   Named tuple with entries t and y and ydot
            ``tstop``   Named tuple with entries t and y and ydot
            ``message`` String with message in case of an error
            =========== ==========================================

        old_api is True : tuple
            tuple with the following elements in order

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    status of the computation (successful or error occured)
            ``t_out``   time, where the solver stopped (when no error occured, t_out == t)
            =========== ==========================================

        """
        return self._integrator.step(t, y_retn, yp_retn)

    def __del__(self):
        """
        Clean up what is needed
        """
        if hasattr(self, '_integrator'):
            del self._integrator

#------------------------------------------------------------------------------
# DAE integrators
#------------------------------------------------------------------------------

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

        ## idas
        try:
            from .sundials import idas
            DaeBase.integrator_classes.append(idas.IDAS)
        except ValueError as msg:
            print('Could not load IDAS solver', msg)
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
