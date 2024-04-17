# -*- coding: utf-8 -*-
# Authors: B. Malengier based on ode.py
r"""
First-order ODE solver
======================

User-friendly interface to various numerical integrators for solving a
system of first order ODEs with prescribed initial conditions:

.. math::

    \frac{dy(t)}{dt} = f(t, y(t)), \quad y(t_0)=y_0

    y(t_0)[i] = y_0[i], i = 0, ..., \mathrm{len}(y_0) - 1

:math:`f(t,y)` is the right hand side function and returns a vector of size
:math:`\mathrm{len}(y_0)`.

"""

from __future__ import print_function

__all__ = ['ode']
__version__ = "$Id$"
__docformat__ = "restructuredtext en"

import re, sys

from numpy import isscalar, array, asarray

class OdeBase(object):
    """
    The interface which ODE solvers must implement.

    Parameters
    ----------
    Rfn : function
        A function which computes the required derivatives. The signature should be ``func(t, y, y_dot, *args, **kwargs)``. Note that *args and **kwargs handling are solver dependent.

    options : mapping
        Additional options for initialization, solver dependent
    """

    integrator_classes = []

    def __init__(self, Rfn, **options):
        raise NotImplementedError('all ODE solvers must implement this')

    def set_options(self, **options):
        """
        Set specific options for the solver.

        Calling set_options a second time, normally resets the solver.
        """
        raise NotImplementedError('all ODE solvers must implement this')

    def solve(self, tspan, y0):
        """
        Runs the solver.

        Parameters
        ----------
        tspan : array (or similar)
            a list of times at which the computed value will be returned. Must
            contain the start time as first entry.

        y0 : array (or similar)
            a list of initial values

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnum)
            ``values``  Named tuple with entries t and y
            ``errors``  Named tuple with entries t and y
            ``roots``   Named tuple with entries t and y
            ``tstop``   Named tuple with entries t and y
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
            ``t_err``  float or None - if recoverable error occured (for example reached maximum number of allowed iterations), this is the time at which it happened
            ``y_err``  numpy array of values corresponding to time t_err
            ========== ==========================================
        """
        raise NotImplementedError('all ODE solvers must implement this')

    def init_step(self, t0, y0):
        """
        Initializes the solver and allocates memory.

        Parameters
        ----------
        t0 : number
            initial time
        y0 : array
            initial condition for y (can be list or numpy array)

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnum)
            ``values``  Named tuple with entries t and y
            ``errors``  Named tuple with entries t and y
            ``roots``   Named tuple with entries t and y
            ``tstop``   Named tuple with entries t and y
            ``message`` String with message in case of an error
            =========== ==========================================

        old_api is True : tuple
            tuple with the following elements in order

            ========== ==========================================
            Field      Meaning
            ========== ==========================================
            ``flag``   boolean status of the computation (successful or error occured)
            ``t_out``  inititial time
            ========== ==========================================
        """
        raise NotImplementedError('all ODE solvers must implement this')

    def step(self, t, y_retn=None):
        """
        Method for calling successive next step of the ODE solver to allow
        more precise control over the solver. The 'init_step' method has to
        be called before the 'step' method.

        A step is done towards time t, and output at t returned.  This time can be higher or lower than the previous time.  If option 'one_step_compute'==True, and the solver supports it, only one internal solver step is done in the direction of t starting at the current step.

        If old_api=True, the old behavior is used: if t>0.0 then integration is performed until this time and results at this time are returned in y_retn if t<0.0 only one internal step is perfomed towards time abs(t) and results after this one time step are returned

        Parameters
        ----------
        t : number

        y_retn : numpy vector (ndim = 1)
            in which the computed value will be stored  (needs to be preallocated).  If None y_retn is not used.

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnum)
            ``values``  Named tuple with entries t and y
            ``errors``  Named tuple with entries t and y
            ``roots``   Named tuple with entries t and y
            ``tstop``   Named tuple with entries t and y
            ``message`` String with message in case of an error
            =========== ==========================================

        old_api is True : tuple
            tuple with the following elements in order

            ========== ==========================================
            Field      Meaning
            ========== ==========================================
            ``flag``   status of the computation (successful or error occured)
            ``t_out``  time, where the solver stopped (when no error occured, t_out == t)
            ========== ==========================================
        """
        raise NotImplementedError('all ODE solvers must implement this')

#------------------------------------------------------------------------------
# User interface
#------------------------------------------------------------------------------

class ode(object):
    """
    A generic interface class to differential equation solvers.


    Parameters
    ----------

    integrator_name : ``'cvode'``, ``'dopri5'`` or ``'dop853'``
        The solver to use.

        ``'cvode'`` selects the CVODE solver from the SUNDIALS package.
        See py:class:`scikits.odes.sundials.cvode.CVODE` for solver specific
        options.

        ``'dopri5'`` selects the Dormand & Prince Runge-Kutta order (4)5
        solver from scipy.

    eqsrhs : right-hand-side function
        Right-hand-side of a first order ode.
        Generally, you can assume the following signature to work:

            eqsrhs(x, y, return_rhs)

        with

            x: independent variable, eg the time, float

            y: array of n unknowns in x

            return_rhs : array that must be updated with the value of the
            right-hand-side, so f(t,y).  The dimension is equal to
            dim(y)

        return value: An integer, 0 for success, 1 for failure.
            It is not guaranteed that a solver takes this status into account

        Some solvers will allow userdata to be passed to eqsrhs, or optional
        formats that are more performant.

    options :  additional options of the solver
        See set_options method of the `integrator_name` you selected for
        details.
        Set option `old_api=False` to use the new API. In the future, this
        will become the default!

    See Also
    --------
    scikits.odes.odeint.odeint : an ODE integrator with a simpler interface
    scipy.integrate : Methods in scipy for ODE integration

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
    >>> def rhseqn(t, x, xdot):
            # we create rhs equations for the problem
            xdot[0] = x[1]
            xdot[1] = - k/m * x[0]

    >>> from scikits.odes import ode
    >>> solver = ode('cvode', rhseqn, old_api=False)
    >>> result = solver.solve([0., 1., 2.], initx)
    >>> print('   t        Solution          Exact')
    >>> print('------------------------------------')
    >>> for t, u in zip(result.values.t, result.values.y):
            print('%4.2f %15.6g %15.6g' % (t, u[0], initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m)))

    More examples in the Examples_ directory and IPython_ worksheets.

    .. _Examples: https://github.com/bmcage/odes/tree/master/docs/src/examples
    .. _IPython: https://github.com/bmcage/odes/tree/master/docs/ipython
    """
    LOADED = False

    def __init__(self, integrator_name, eqsrhs, **options):
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

        Calling set_options a second time, is only possible for options that
        can change during runtime.
        """
        return self._integrator.set_options(**options)

    def solve(self, tspan, y0):
        """
        Runs the solver.

        Parameters
        ----------
        tspan : array (or similar)
            a list of times at which the computed value will be returned. Must
            contain the start time as first entry.

        y0 : array (or similar)
            a list of initial values

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnum)
            ``values``  Named tuple with entries t and y
            ``errors``  Named tuple with entries t and y
            ``roots``   Named tuple with entries t and y
            ``tstop``   Named tuple with entries t and y
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
            ``t_err``  float or None - if recoverable error occured (for example reached maximum number of allowed iterations), this is the time at which it happened
            ``y_err``  numpy array of values corresponding to time t_err
            ========== ==========================================

        """
        return self._integrator.solve(tspan, y0)

    def init_step(self, t0, y0):
        """
        Initializes the solver and allocates memory. It is not needed to
        call this method if solve is used to compute the solution. In the case
        step is used, init_step must be called first.

        Parameters
        ----------
        t0 : number
            initial time
        y0 : array
            initial condition for y (can be list or numpy array)

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnum)
            ``values``  Named tuple with entries t and y
            ``errors``  Named tuple with entries t and y
            ``roots``   Named tuple with entries t and y
            ``tstop``   Named tuple with entries t and y
            ``message`` String with message in case of an error
            =========== ==========================================

        old_api is True : tuple
            tuple with the following elements in order

            ========== ==========================================
            Field      Meaning
            ========== ==========================================
            ``flag``   boolean status of the computation (successful or error occured)
            ``t_out``  inititial time
            ========== ==========================================

        """
        return self._integrator.init_step(t0, y0)

    def step(self, t, y_retn=None):
        """
        Method for calling successive next step of the ODE solver to allow
        more precise control over the solver. The 'init_step' method has to
        be called before the 'step' method.

        A step is done towards time t, and output at t returned.  This time can be higher or lower than the previous time.  If option 'one_step_compute'==True, and the solver supports it, only one internal solver step is done in the direction of t starting at the current step.

        If old_api=True, the old behavior is used: if t>0.0 then integration is performed until this time and results at this time are returned in y_retn if t<0.0 only one internal step is perfomed towards time abs(t) and results after this one time step are returned

        Parameters
        ----------
        t : number

        y_retn : numpy vector (ndim = 1)
            in which the computed value will be stored  (needs to be preallocated).  If None y_retn is not used.

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnum)
            ``values``  Named tuple with entries t and y
            ``errors``  Named tuple with entries t and y
            ``roots``   Named tuple with entries t and y
            ``tstop``   Named tuple with entries t and y
            ``message`` String with message in case of an error
            =========== ==========================================

        old_api is True : tuple
            tuple with the following elements in order

            ========== ==========================================
            Field      Meaning
            ========== ==========================================
            ``flag``   status of the computation (successful or error occured)
            ``t_out``  time, where the solver stopped (when no error occured, t_out == t)
            ========== ==========================================

        """
        return self._integrator.step(t, y_retn)

    def set_tstop(self, tstop):
        """
        Add a stop time to the integrator past which he is not allowed to
        integrate.

        Parameters
        ----------
        tstop : float time
            Time point in the future where the integration must stop. You can
            indicate like this that integration past this point is not allowed,
            in order to avoid undefined behavior.
            You can achieve the same result with a call to
            `set_options(tstop=tstop)`
        """
        if hasattr(self._integrator, 'set_tstop'):
            self._integrator.set_tstop(tcrit)
        else:
            self._integrator.set_options(tstop=tstop)

    def get_info(self):
        """
        Return additional information about the state of the integrator.

        Returns
        -------

        A dictionary filled with internal data as exposed by the integrator.
        See the `get_info` method of your chosen integrator for details.

        """
        if hasattr(self._integrator, 'get_info'):
            return self._integrator.get_info()
        else:
            return {}

#------------------------------------------------------------------------------
# ODE integrators
#------------------------------------------------------------------------------

def find_ode_integrator(name):
    if not ode.LOADED:
        ## cvode
        try:
            from .sundials import cvode
            OdeBase.integrator_classes.append(cvode.CVODE)
        except ValueError as msg:
            print('Could not load CVODE solver', msg)
        except ImportError:
            print(sys.exc_info()[1])
        ## cvodes
        try:
            from .sundials import cvodes
            OdeBase.integrator_classes.append(cvodes.CVODES)
        except ValueError as msg:
            print('Could not load CVODES solver', msg)
        except ImportError:
            print(sys.exc_info()[1])

        ## dopri5 and dop853
        try:
            from .dopri5 import dopri5, dop853
        except ImportError:
            print(sys.exc_info()[1])

        ode.LOADED = True

    for cl in OdeBase.integrator_classes:
        if re.match(name, cl.__name__, re.I):
            return cl
        elif hasattr(cl, name) and re.match(name, cl.name, re.I):
            return cl
    raise ValueError('Integrator name %s does not exist' % name)
