# -*- coding: utf-8 -*-
# Created on Thu Jan 28 14:59:25 2016
# @author: benny
"""
Making scipy ode solvers available via the ode API
==================================================

dopri5
------
    This is an explicit runge-kutta method of order (4)5 due to Dormand & Prince
    (with stepsize control and dense output).
    The API of this solver is as the other scikit.odes ODE solvers

    Authors:

        E. Hairer and G. Wanner Universite de Geneve, Dept. de Mathematiques CH-1211 Geneve 24, Switzerland e-mail: ernst.hairer@math.unige.ch, gerhard.wanner@math.unige.ch

    This code is described in [HNW93].

    This integrator accepts the following options:

        atol : float or sequence absolute tolerance for solution, default 1e-12
        rtol : float or sequence relative tolerance for solution, default 1e-6
        nsteps : int Maximum number of (internally defined) steps allowed during one call to the solver. Default=500
        first_step : float
        max_step : float
        safety : float Safety factor on new step selection (default 0.9)
        ifactor : float
        dfactor : float Maximum factor to increase/decrease step size by in one step
        beta : float Beta parameter for stabilised step size control.
        verbosity : int Switch for printing messages (< 0 for no messages).

References
[HNW93]	(1, 2) E. Hairer, S.P. Norsett and G. Wanner, Solving Ordinary Differential Equations i. Nonstiff Problems. 2nd edition. Springer Series in Computational Mathematics, Springer-Verlag (1993)

dop853
------
    This is an explicit runge-kutta method of order 8(5,3) due to Dormand & Prince
    (with stepsize control and dense output).
    Options and references the same as “dopri5”.

"""

from __future__ import print_function

from collections import namedtuple
from enum import IntEnum
from warnings import warn

import numpy as np

from .ode import OdeBase


SolverReturn = namedtuple(
    "SolverReturn", [
        "flag", "values", "errors", "roots",
        "tstop", "message"
    ]
)

SolverVariables = namedtuple("SolverVariables", ["t", "y"])

class StatusEnumDOP(IntEnum):
    SUCCESS = 1
    SOLOUT  = 2

    INPUT_FAIL     = -1
    NMAX_FAIL      = -2
    STEPSIZE_FAIL  = -3
    STIFFNESS_FAIL = -4

    UNEXPECTED_IDID = -10

STATUS_MESSAGE = {
    StatusEnumDOP.SUCCESS: 'computation successful',
    StatusEnumDOP.SOLOUT: 'comput. successful (interrupted by solout)',
    StatusEnumDOP.INPUT_FAIL: 'input is not consistent',
    StatusEnumDOP.NMAX_FAIL: 'larger nmax is needed',
    StatusEnumDOP.STEPSIZE_FAIL: 'step size becomes too small',
    StatusEnumDOP.STIFFNESS_FAIL: 'problem is probably stiff (interrupted)',
    StatusEnumDOP.UNEXPECTED_IDID: 'Unexpected idid, check warnings for info',
}

class DOPSolveException(Exception):
    """Base class for exceptions raised by `DOP.validate_flags`."""
    def __init__(self, soln):
        self.soln = soln
        self.args = (self._message.format(soln),)

class DOPSolveFailed(DOPSolveException):
    """`DOP.solve` failed to reach endpoint"""
    _message = (
        "Solver failed with flag {0.flag} and finished at {0.errors.t}"
        "with values {0.errors.y}."
    )

class dopri5(OdeBase):
    try:
        from scipy.integrate import ode as _runner
    except ImportError:
        print(sys.exc_info()[1])
        _runner = None

    name = 'dopri5'

    default_values = {
            'rtol': 1e-6,
            'atol': 1e-12,
            'nsteps': 500,
            'max_step': 0.0,
            'first_step': 0.0,  # determined by solver
            'safety': 0.9,
            'ifactor': 10.0,
            'dfactor': 0.2,
            'beta': 0.0,
            'verbosity': -1, # no messages if negative
            }

    def __init__(self, Rfn, **options):
        """
        Initialize the ODE Solver and it's default values

        Parameters
        ----------
            Rfn     - right-hand-side function
            options - additional options for initialization
        """
        self.options = self.default_values
        self.set_options(rfn=Rfn, **options)
        self._validate_flags = None
        self.solver = None
        self.initialized = False

    def set_options(self, **options):
        """
        Set specific options for the solver.

        Calling set_options a second time, normally resets the solver.
        """
        for (key, value) in options.items():
            self.options[key.lower()] = value
        self.initialized = False

    def _init_data(self):
        self.rfn = self.options['rfn']
        self.N = len(self.y0)

    def _wrap_Rfn(self, t, y):
        yout = np.empty(self.N, float)
        self.rfn(t, y, yout)
        return yout

    def init_step(self, t0, y0):
        """
        Initializes the solver and allocates memory.

        Parameters
        ----------
            t0     - initial time
            y0     - initial condition for y (can be list or numpy array)

        Returns
        -------
         if old_api:
            not supported

         if old_api False:
            A named tuple, with entries:
                flag   = An integer flag (StatusEnumDop)
                values = Named tuple with entries t and y and ydot. y will
                            correspond to y_retn value and ydot to yp_retn!
                errors = Named tuple with entries t_err and y_err
                roots  = Named tuple with entries t_roots and y_roots
                tstop  = Named tuple with entries t_stop and y_tstop
                message= String with message in case of an error

        """
        self.y0 = y0
        self.t = t0
        self._init_data()

        self.solver = self._runner(self._wrap_Rfn)\
                        .set_integrator(self.name,
                            rtol = self.options['rtol'],
                            atol = self.options['atol'],
                            nsteps = self.options['nsteps'],
                            max_step = self.options['max_step'],
                            first_step = self.options['first_step'],  # determined by solver
                            safety = self.options['safety'],
                            ifactor = self.options['ifactor'],
                            dfactor = self.options['dfactor'],
                            beta = self.options['beta'],
                            verbosity = self.options['verbosity'])\
                        .set_initial_value(y0, t0)
        self.initialized = True
        y_retn  = np.empty(np.alen(y0), float)
        y_retn[:] = y0[:]
        soln = SolverReturn(
            flag=StatusEnumDOP.SUCCESS,
            values=SolverVariables(t=t0, y=y_retn),
            errors=SolverVariables(t=None, y=None),
            roots=SolverVariables(t=None, y=None),
            tstop=SolverVariables(t=None, y=None),
            message=STATUS_MESSAGE[StatusEnumDOP.SUCCESS]
        )
        if self._validate_flags:
            return self.validate_flags(soln)
        return soln

    def step(self, t, y_retn=None):
        """
        Method for calling successive next step of the ODE solver to allow
        more precise control over the solver. The 'init_step' method has to
        be called before the 'step' method.

        Parameters
        ----------
            t - A step is done towards time t, and output at t returned.
                This time can be higher or lower than the previous time.
                If option 'one_step_compute'==True, and the solver supports
                it, only one internal solver step is done in the direction
                of t starting at the current step.

                If old_api=True, the old behavior is used:
                 if t>0.0 then integration is performed until this time
                          and results at this time are returned in y_retn
                 if t<0.0 only one internal step is perfomed towards time abs(t)
                         and results after this one time step are returned
            y_retn - numpy vector (ndim = 1) in which the computed
                     value will be stored  (needs to be preallocated).  If
                     None y_retn is not used.
        Returns
        -------
         if old_api:
             not supported

         if old_api False:
            A named tuple, with entries:
                flag   = An integer flag (StatusEnumDOP)
                values = Named tuple with entries t and y. y will
                            correspond to y_retn value
                errors = Named tuple with entries t_err and y_err
                roots  = Named tuple with entries t_roots and y_roots
                tstop  = Named tuple with entries t_stop and y_tstop
                message= String with message in case of an error

        """
        if not self.initialized == False:
            raise ValueError("DOPRI:step: init_step must be run before running the step method.")
        if t <= self.t:
            raise ValueError("Integration must be forward! t must be > previous timestep")
        y = self.solver.integrate(t)
        y_err = None
        t_err = None
        if not self.solver.successful():
            flag = StatusEnumDOP.UNEXPECTED_IDID
            y_err = y
            t_err = t
        else:
            flag = StatusEnumDOP.SUCCESS
            self.y = y
            self.t = self.solver.t

        return SolverReturn(
            flag=flag,
            values=SolverVariables(t=self.t, y=y),
            errors=SolverVariables(t=t_err, y=y_err),
            roots=SolverVariables(t=None, y=None),
            tstop=SolverVariables(t=None, y=None),
            message=STATUS_MESSAGE[flag]
        )

    def solve(self, tspan, y0):
        """
        Runs the solver.

        Parameters
        ----------
            tspan - an list/array of times at which the computed value will be
                    returned. Must contain the start time as first entry..
            y0    - list/numpy array of initial values

        Returns
        -------
         if old_api
            Not supported
         if old_api False:
            A named tuple, with entries:
                flag   = An integer flag
                values = Named tuple with entries t and y
                errors = Named tuple with entries t and y
                roots  = Named tuple with entries t and y
                tstop  = Named tuple with entries t and y
                message= String with message in case of an error
        """
        self.initialized = False
        soln = self.init_step(tspan[0], y0)
        nrt = len(tspan)
        t_retn  = np.empty(np.shape(tspan), float)
        y_retn  = np.empty([np.alen(tspan), np.alen(y0)], float)

        t_retn[0] = tspan[0]
        y_retn[0, :] = y0

        idx = 1
        while self.solver.successful() and idx<nrt:
            time = tspan[idx]
            y = self.solver.integrate(time)
            t_retn[idx] = time
            y_retn[idx, :] = y
            idx += 1

        y_err = None
        t_err = None
        if not self.solver.successful():
            flag = StatusEnumDOP.UNEXPECTED_IDID
            y_err = y_retn[idx-1]
            t_err = t_retn[idx-1]
            # return values computed so far
            t_retn  = t_retn[0:idx-1]
            y_retn  = y_retn[0:idx-1, :]
        else:
            flag = StatusEnumDOP.SUCCESS

        soln = SolverReturn(
            flag=flag,
            values=SolverVariables(t=t_retn, y=y_retn),
            errors=SolverVariables(t=t_err, y=y_err),
            roots=SolverVariables(t=None, y=None),
            tstop=SolverVariables(t=None, y=None),
            message=STATUS_MESSAGE[flag]
        )
        if self._validate_flags:
            return self.validate_flags(soln)
        return soln

    def validate_flags(self, soln):
        """
        Validates the flag returned by `dopri.solve`.

        Validation happens using the following scheme:
         * failures (`flag` < 0) raise `DOPSolveFailed` or a subclass of it;
         * otherwise, return an instance of `SolverReturn`.

        """
        if soln.flag == StatusEnumDOP.SUCCESS:
            return soln
        if soln.flag < 0:
            raise DOPSolveFailed(soln)
        warn(WARNING_STR.format(soln.flag, *soln.err_values))
        return soln

class dop853(dopri5):
    name = 'dop853'
    default_values = {
            'rtol': 1e-6,
            'atol': 1e-12,
            'nsteps': 500,
            'max_step': 0.0,
            'first_step': 0.0,  # determined by solver
            'safety': 0.9,
            'ifactor': 6.0,
            'dfactor': 0.3,
            'beta': 0.0,
            'verbosity': -1, # no messages if negative
            }


if dopri5._runner:
    OdeBase.integrator_classes.append(dopri5)

if dop853._runner:
    OdeBase.integrator_classes.append(dop853)
