# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:38:24 2016

@author: benny
"""

from __future__ import print_function

from copy import copy
from .ode import ode

def odeint(rhsfun, tout, y0,  method='bdf', **options):
    """
    Integrate a system of ordinary differential equations.
    *odeint* is a wrapper around the ode class, as a convenience function to
    quickly integrate a system of ode.
    Solves the initial value problem for stiff or non-stiff systems
    of first order ode's:

        rhs = dy/dt = fun(t, y)

    where y can be a vector, then rhsfun must be a function computing rhs with
    signature:

        rhsfun(t, y, rhs)

    storing the computed dy/dt in the rhs array passed to the function

    Parameters
    ----------
    rhsfun : callable(t, y, out)
        Computes the derivative at dy/dt in terms of t and y, and stores in out
    y0 : array
        Initial condition on y (can be a vector).
    t : array
        A sequence of time points for which to solve for y.  The initial
        value point should be the first element of this sequence.
    method : string, solution method to use.
        Available are all the ode class solvers as well as some convenience
        shorthands:

        =======  ==============================================================
        Method   Meaning
        =======  ==============================================================
        bdf      This uses the 'cvode' solver in default from, which is a
                 variable step, variable coefficient Backward Differentiation
                 Formula solver, good for stiff ODE. Newton iterations are
                 used to solve the nonlinear system.
        admo     This uses the 'cvode' solver with option lmm_type='ADAMS',
                 which is a variable step Adams-Moulton method (linear
                 multistep method), good for non-stiff ODE. Functional
                 iterations are used to solve the nonlinear system.
        rk5      This uses the 'dopri5' solver, which is a variable step
                 Runge-Kutta method of order (4)5 (use for non-stiff ODE)
        rk8      This uses the 'dop853' solver, which is a variable step
                 Runge-Kutta method of order 8(5,3)
        =======  ==============================================================

        For educational purposes, you can also access following methods:

        =======  ==============================================================
        Method   Meaning
        =======  ==============================================================
        beuler   This is the Implicit Euler (backward Euler) method (order 1),
                 which is obtained via the 'bdf' method, setting the order
                 option to 1, setting large tolerances,  and fixing the
                 stepsize.
                 Use option 'step' to change stepsize, default: step=0.05.
                 Use option 'rtol' and 'atol' to use more strict tolerances
                 Note: this is not completely the backward Euler method, as
                 the cvode solver has added control options!
        trapz    This is the Trapezoidal Rule method (order 2), which is
                 obtained via the 'admo' method, setting option order to 2,
                 setting large tolerances and fixing the stepsize.
                 Use option 'step' to change stepsize, default: step=0.05.
                 Use option 'rtol' and 'atol' to use more strict tolerances
                 Note: The cvode solver might change the order to 1 internally
                 in which case this becomes beuler method. Set atol, rtol
                 options as strict as possible.
        =======  ==============================================================

        You can also access the solvers of ode via their names:

        =======  ==============================================================
        Method   Meaning
        =======  ==============================================================
        cvode    This uses the 'cvode' solver
        dopri5   This uses the 'dopri5' solver
        dop853   This uses the 'dop853' solver
        =======  ==============================================================

    options : extra solver options, optional
        Every solver has it's own extra options, see the ode class and the
        details of the solvers available there to know the options possible per
        solver

    Returns
    -------
    solution : named tuple
        A single named tuple is returned containing the result of the
        integration.

        ========  ==========================================
        Field     Meaning
        ========  ==========================================
        flag      An integer flag
        values    Named tuple with fields t and y
        errors    Named tuple with fields t and y
        roots     Named tuple with fields t and y
        tstop     Named tuple with fields t and y
        message   String with message in case of an error
        ========  ==========================================

    See Also
    --------
    scikits.odes.ode.ode : a more object-oriented integrator
    scikits.odes.dae.dae : a solver for differential-algebraic equations
    scipy.integrate.quad : for finding the area under a curve

    Examples
    --------
    The second order differential equation for the angle `theta` of a
    pendulum acted on by gravity with friction can be written:

    .. math:: \\theta''(t) + b \\theta'(t) + c \\sin(\\theta(t)) = 0

    where `b` and `c` are positive constants, and a prime (') denotes a
    derivative.  To solve this equation with `odeint`, we must first convert
    it to a system of first order equations.  By defining the angular
    velocity ``omega(t) = theta'(t)``, we obtain the system:

    .. math:: \\theta'(t) = \\omega(t)
     \\omega'(t) = -b \\omega(t) - c \\sin(\\theta(t))

    We assume the constants are `b` = 0.25 and `c` = 5.0:

    >>> b = 0.25
    >>> c = 5.0

    Let `y` be the vector [`theta`, `omega`].  We implement this system
    in python as:

    >>> def pend(t, y, out):
     ...     theta, omega = y
     ...     out[:] = [omega, -b*omega - c*np.sin(theta)]
     ...

    In case you want b and c easily changable, make pend a class method, and
    consider attributes b and c accessible via `self.b` and `self.c`.
    For initial conditions, we assume the pendulum is nearly vertical
    with `theta(0)` = `pi` - 0.1, and it initially at rest, so
    `omega(0)` = 0.  Then the vector of initial conditions is

    >>> y0 = [np.pi - 0.1, 0.0]

    We generate a solution 101 evenly spaced samples in the interval
    0 <= `t` <= 10.  So our array of times is

    >>> t = np.linspace(0, 10, 101)

    Call `odeint` to generate the solution.

    >>> from scikits.odes.odeint import odeint
    >>> sol = odeint(pend, t, y0)

    The solution is a named tuple `sol`. sol.values.y is an array with shape (101, 2).
    The first column is `theta(t)`, and the second is `omega(t)`.
    The following code plots both components.

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(sol.values.t, sol.values.y[:, 0], 'b', label='theta(t)')
    >>> plt.plot(sol.values.t, sol.values.y[:, 1], 'g', label='omega(t)')
    >>> plt.legend(loc='best')
    >>> plt.xlabel('t')
    >>> plt.grid()
    >>> plt.show()
    """
    t = copy(tout)
    y0 = copy(y0)

    int_name = 'cvode'
    #always use old api
    options['old_api'] = False
    if (method == 'bdf'):
        int_name = 'cvode'
    elif (method == 'admo'):
        options['lmm_type'] = 'ADAMS'
        options['nonlinsolver'] = 'fixedpoint'
        int_name = 'cvode'
    elif (method == 'rk5'):
        int_name = 'dopri5'
    elif (method == 'rk8'):
        int_name = 'dop853'
    elif (method == 'beuler' or method == 'trapz'):
        int_name = 'cvode'
        options['order'] = 1
        if not options.has_key('step'):
            options['step'] = 0.05
        options['min_step_size'] = options['step']
        options['max_step_size'] = options['step']
        #no error control possible, try to disable it:
        if not options.has_key('atol'):
            options['atol'] = 1e3
        if not options.has_key('rtol'):
            options['rtol'] = 1e3
        del options['step']
        if (method == 'trapz'):
            options['lmm_type'] = 'ADAMS'
            options['nonlinsolver'] = 'fixedpoint'
            options['order'] = 2
    else:
        int_name = method

    solver = ode(int_name, rhsfun, **options)
    return solver.solve(t, y0)
