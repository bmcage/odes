# cython: embedsignature=True
from cpython.exc cimport PyErr_CheckSignals
from collections import namedtuple
from enum import IntEnum
import inspect
from warnings import warn

include "sundials_config.pxi"

import numpy as np
cimport numpy as np


from .c_sundials cimport realtype, N_Vector
from .c_nvector_serial cimport *
from .c_sunmatrix cimport *
from .c_sunlinsol cimport *
from .c_sunnonlinsol cimport *

from .ida cimport (IDA_RhsFunction, IDA_WrapRhsFunction, IDA_RootFunction,
                   IDA_WrapRootFunction, IDA_JacRhsFunction,
                   IDA_WrapJacRhsFunction, IDA_PrecSetupFunction,
                   IDA_WrapPrecSetupFunction, IDA_PrecSolveFunction,
                   IDA_WrapPrecSolveFunction, IDA_JacTimesVecFunction,
                   IDA_WrapJacTimesVecFunction, IDA_JacTimesSetupFunction,
                   IDA_WrapJacTimesSetupFunction, IDA_ContinuationFunction,
                   IDA_ErrHandler, IDA_WrapErrHandler, IDA_data,
                   IDA)

from .c_idas cimport *
from .common_defs cimport (
    nv_s2ndarray, ndarray2nv_s, ndarray2SUNMatrix, DTYPE_t, INDEX_TYPE_t,
)
from .common_defs import DTYPE, INDEX_TYPE
# this is needed because we want DTYPE and INDEX_TYPE to be
# accessible from python (not only in cython)
from . import (
    IDASolveFailed, IDASolveFoundRoot, IDASolveReachedTSTOP, _get_num_args,
)


# TODO: parallel implementation: N_VectorParallel
# TODO: linsolvers: check the output value for errors
# TODO: optimize code for compiler

SolverReturn = namedtuple(
    "SolverReturn", [
        "flag", "values", "errors", "roots",
        "tstop", "message"
    ]
)

SolverVariables = namedtuple("SolverVariables", ["t", "y", "ydot"])

class StatusEnumIDA(IntEnum):
    SUCCESS           = IDA_SUCCESS           # 0
    TSTOP_RETURN      = IDA_TSTOP_RETURN      # 1
    ROOT_RETURN       = IDA_ROOT_RETURN       # 2

    WARNING           = IDA_WARNING           # 99

    TOO_MUCH_WORK     = IDA_TOO_MUCH_WORK     # -1
    TOO_MUCH_ACC      = IDA_TOO_MUCH_ACC      # -2
    ERR_FAIL          = IDA_ERR_FAIL          # -3
    CONV_FAIL         = IDA_CONV_FAIL         # -4

    LINIT_FAIL        = IDA_LINIT_FAIL        # -5
    LSETUP_FAIL       = IDA_LSETUP_FAIL       # -6
    LSOLVE_FAIL       = IDA_LSOLVE_FAIL       # -7
    RES_FAIL          = IDA_RES_FAIL          # -8
    REP_RES_ERR       = IDA_REP_RES_ERR       # -9
    RTFUNC_FAIL       = IDA_RTFUNC_FAIL       # -10
    CONSTR_FAIL       = IDA_CONSTR_FAIL       # -11

    FIRST_RES_FAIL    = IDA_FIRST_RES_FAIL    # -12
    LINESEARCH_FAIL   = IDA_LINESEARCH_FAIL   # -13
    NO_RECOVERY       = IDA_NO_RECOVERY       # -14
    NLS_INIT_FAIL     = IDA_NLS_INIT_FAIL     # -15
    NLS_SETUP_FAIL    = IDA_NLS_SETUP_FAIL    # -16
    NLS_FAIL          = IDA_NLS_FAIL          # -17

    MEM_NULL          = IDA_MEM_NULL          # -20
    MEM_FAIL          = IDA_MEM_FAIL          # -21
    ILL_INPUT         = IDA_ILL_INPUT         # -22
    NO_MALLOC         = IDA_NO_MALLOC         # -23
    BAD_EWT           = IDA_BAD_EWT           # -24
    BAD_K             = IDA_BAD_K             # -25
    BAD_T             = IDA_BAD_T             # -26
    BAD_DKY           = IDA_BAD_DKY           # -27
    VECTOROP_ERR      = IDA_VECTOROP_ERR      # -28

    UNRECOGNIZED_ERROR= IDA_UNRECOGNIZED_ERROR# -99

STATUS_MESSAGE = {
    StatusEnumIDA.SUCCESS: "Successful function return.",
    StatusEnumIDA.TSTOP_RETURN: "Reached specified stopping point",
    StatusEnumIDA.ROOT_RETURN: "Found one or more roots",
    StatusEnumIDA.WARNING: "Succeeded but something unusual happened",
    StatusEnumIDA.TOO_MUCH_WORK: "Could not reach endpoint",
    StatusEnumIDA.TOO_MUCH_ACC: "Could not satisfy accuracy",
    StatusEnumIDA.ERR_FAIL: "Error test failures occurred too many times during one internal time step or minimum step size was reached.",
    StatusEnumIDA.CONV_FAIL: "Convergence test failures occurred too many times during one internal time step or minimum step size was reached.",
    StatusEnumIDA.LINIT_FAIL: "The linear solver’s initialization function failed.",
    StatusEnumIDA.LSETUP_FAIL: "The linear solver’s setup function failed in an unrecoverable manner.",
    StatusEnumIDA.LSOLVE_FAIL: "The linear solver’s solve function failed in an unrecoverable manner.",
    StatusEnumIDA.RES_FAIL: "The residual function had a non-recoverable error",
    StatusEnumIDA.REP_RES_ERR: "There are repeated recoverable residual errors.",
    StatusEnumIDA.RTFUNC_FAIL: "The rootfinding routine failed in an unrecoverable manner.",
    StatusEnumIDA.CONSTR_FAIL: "The inequality constraints could not be met.",
    StatusEnumIDA.FIRST_RES_FAIL: "The residual function failed at the first call.",
    StatusEnumIDA.LINESEARCH_FAIL: "The linesearch failed (on steptol test)",
    StatusEnumIDA.NO_RECOVERY: "The residual routine or the linear setup or solve routine had a recoverable error, but IDACalcIC was unable to recover.",
    StatusEnumIDA.NLS_INIT_FAIL: "The nonlinear solver's init routine failed.",
    StatusEnumIDA.NLS_SETUP_FAIL: "The nonlinear solver setup failed unrecoverably.",
    StatusEnumIDA.NLS_FAIL: "The nonlinear solver failed in an unrecoverable manner",
    StatusEnumIDA.MEM_NULL: "Integrator memory is NULL.",
    StatusEnumIDA.MEM_FAIL: "A memory request failed.",
    StatusEnumIDA.ILL_INPUT: "Invalid input detected.",
    StatusEnumIDA.NO_MALLOC: "Attempt to call before IDAMalloc. Allocate memory first.",
    StatusEnumIDA.BAD_EWT: "Some initial ewt component = 0.0 illegal.",
    StatusEnumIDA.BAD_K: "Illegal value for k. If the requested k is not in the range 0,1,...,order used ",
    StatusEnumIDA.BAD_T: "Illegal value for t. If t is not within the interval of the last step taken.",
    StatusEnumIDA.BAD_DKY: "The dky vector is NULL",
    StatusEnumIDA.VECTOROP_ERR: "Vector operation error",
    StatusEnumIDA.UNRECOGNIZED_ERROR: "Unrecognized error",

}

WARNING_STR = "Solver succeeded with flag {} and finished at {} with values {}"

cdef class IDAS_data(IDA_data):
    def __cinit__(self, N, Ns=0):
        super(IDAS_data, self).__init__(N)
        
        self.yS_tmp = np.empty(Ns, DTYPE)
        self.ySdot_tmp = np.empty(Ns, DTYPE)

cdef class IDAS(IDA):

    def __cinit__(self, Rfn, **options):
        """
        Initialize the IDA Solver and it's default values

        Input:
            Rfn     - residual function
            options - additional options for initialization
        """
        super(IDAS, self).__init__(Rfn, **options)
        
        self.Ns       = -1

    def set_options(self, **options):
        """
        Reads the options list and assigns values for the solver.

        All options list:
            'verbosity':
                Values: 0,1,2,...
                Description:
                    Set the level of verbosity. The higher number user, the
                    more verbose the output will be. Default is 1.
                    Deprecated, does nothing!
            'implementation':
                Values: 'serial' (= default), 'parallel'
                Description:
                    Using serial or parallel implementation of the solver.
                    #TODO: curently only serial implementation is working
            'use_relaxation':
                Values: False (= default), True
                Description:
                    Uses relaxation algorithm for solving (if possible).
            'rtol':
                Values: float,  1e-6 = default
                Description:
                    Relative tolerancy.
            'atol':
                Values: float or numpy array of floats,  1e-12 = default
                Description:
                    Absolute tolerancy
            'linsolver':
                Values: 'dense' (= default), 'lapackdense', 'band',
                        'lapackband', 'spgmr', 'spbcg', 'sptfqmr'
                Description:
                    Specifies used linear solver.
                    Limitations: Linear solvers for dense and band matrices can
                        be used only for serial implementation. For parallel
                        implementation use lapackdense or lapackband
                         respectively.
            'lband', 'uband':
                Values: non-negative integer, 0 = default
                Description:
                    Specifies the width of the lower band (below the main
                    diagonal) and/or upper diagonal (above the main diagonal).
                    So the total band width is lband + 1 + uband, where the 1
                    stands for the main diagonal (specially, lband = uband = 0
                    [that is the default] denotes the band width = 1, i.e. the
                    main diagonal only). Used only if 'linsolver' is band.
            'maxl':
                Values: 0 (= default), 1, 2, 3, 4, 5
                Description:
                    Dimension of the number of used Krylov subspaces
                    (used only by 'spgmr', 'spbcg', 'sptfqmr' linsolvers)
            'tstop':
                Values: float, 0.0 = default
                Description:
                    Maximum time until which we perform the computations.
                    Default is 0.0. Once the value is set 0.0, it cannot be
                    disable (but it will be automatically disable when tstop
                    is reached and has to be reset again in next run).
            'order':
                Values: 1, 2, 3, 4, 5 (= default)
                Description:
                    Specifies the maximum order of the linear multistep method.
            'max_steps':
                Values: positive integer, 0 = default (uses value of 500)
                Description:
                    Maximum number of (internally defined) steps allowed during
                    one call to the solver.
            'max_step_size':
                Values: non-negative float, 0.0 = default
                Description:
                    Restricts the maximal (absolute) internal step value taken
                    by the solver. The value of 0.0 uses no restriction on the
                    maximal step size.
            'first_step_size':
                Values: float , 0.0 = default
                Description:
                    Sets the first step size. DAE solver can suffer on the
                    first step, so set this to circumvent this. The value of 0.0
                    uses the solver's internal default value.
            'compute_initcond':
                Values: 'y0', 'yp0', None
                Description:
                    Specifies what initial condition is calculated for given
                    problem.

                    'y0'  - calculates/improves the y0 initial condition
                            (considering yp0 to be accurate)
                    'yp0' - calculates/improves the yp0 initial condition
                            (considering y0 to be accurate)
                    None  - don't compute any initial conditions
                            (user provided values for y0 an yp0 are considered
                             to be consistent and accurate)
            'compute_initcond_t0':
                Values: float, 0.01 = default
                Description:
                    When calculating the initial condition, specifies the time
                    until which the solver tries to
                    get the consistent values for either y0 or yp0 relative to
                    the starting time. Positive if t1 > t0, negative if t1 < t0
            'one_step_compute':
                Values: False (default) or True
                Description:
                    Only influences the step() method.
                    Solver computes normally up to the required output time.
                    If value is True, solver insteads does one internal step
                    of the solver in the direction of the output time, as seen
                    from the current time of the solution.
            'user_data':
                Values: python object, None = default
                Description:
                    Additional data that are supplied to each call of the
                    residual function 'Rfn' (see below), root function 'rootfn'
                    and Jacobi function 'jacfn' (if specified one).
            'Rfn':
                Values: function of class IDA_RhsFunction or a python function
                    with a signature (t, y, yp, resultout)
                Description:
                    Defines the residual function (which has to be a subclass of
                    IDA_RhsFunction class, or a normal python function with
                    signature (t, y, yp, resultout) ).
                    This function takes as input arguments current time t,
                    current value of y, yp, numpy array of returned residual
                    and optional userdata. Return value is 0 if successfull.
                    This option is mandatory.
            'rootfn':
                Values: function of class IDA_RootFunction or a python function
                    with signature (t, y, yp, g, user_data)
                Description:
                    Defines a function that fills a vector 'g' with values
                    (g is a vector of size k). Vector 'g represents conditions
                    which when satisfied, the solver stops. Namely, the solver
                    stops when condition g[i] is zero for some i.
                    For 'user_data' argument see 'user_data' option
                    above.
            'nr_rootfns':
                Value: integer
                Description:
                    The length of the array returned by 'rootfn' (see above),
                    i.e. number of conditions for which we search the value 0.
            'jacfn':
                Values: function of class IDA_JacRhFunction or python function
                Description:
                    Defines the jacobian of the residual function as
                                   dres/dy + cj dres/dyp,
                    and has to be a subclass of IDA_JacRhFunction class or
                    a python function.
                    This function takes as input arguments current time t,
                    current value of y, yp, cj, and 2D numpy array of returned
                    jacobian
                    and optional userdata. Return value is 0 if successfull.
                    Jacobian is only used for dense or lapackdense linear solver
            'algebraic_vars_idx':
                Values: numpy vector or None (= default)
                Description:
                    If the given problem is of type DAE, some items of the residual
                    vector returned by the 'resfn' have to be treated as
                    algebraic equations, and algebraic variables must be defined.
                    These algebraic variables are denoted by the position (index)
                    in the state vector y.
                    All these indexes have to be specified in the
                    'algebraic_vars_idx' array.
            'exclude_algvar_from_error':
                Values: False (= default), True
                Description:
                    Indicates whether or not to suppress algebraic variables in
                    the local error test. If 'algebraic_vars_idx'
                    vector not specified, this value is ignored.

                    The use of this option (i.e. set to True) is discouraged
                    when solving DAE systems of index 1, whereas it
                    is generally encouraged for systems of index 2 or more.
            'constraints_idx':
                Values: numpy vector or None (= default)
                Description:
                    Constraint the individual variables. The variables are
                    denoted by the position (index) in the residual vector.
                    All these indexes are/have to be specified in this
                    'constraints_idx' array.
            'constraints_type':
                Values: numpy vector or None (= default)
                Description:
                    The actuall type of contraints applied to the of the
                    variables specified by 'constraints_idx'. The type is
                    one of:  0.0 - no contraint
                             1.0 - variable has to be non-negative (i.e. >= 0)
                             2.0 - variable has to be positive (i.e. > 0)
                            -1.0 - variable has to be non-positive (i.e. <= 0)
                            -2.0 - variable has to be negative (i.e. < 0)
            'precond_type':
                default = None
            'prec_setupfn':
                Values: function of class IDA_PrecSetupFunction
                Description:
                    Defines a function that setups the preconditioner on change
                    of the Jacobian. This function takes as input arguments
                    current time t, current value of y, flag jok that indicates
                    whether Jacobian-related data has to be updated, flag jcurPtr
                    that should be set to True (jcurPtr.value = True) if Jacobian
                    data was recomputed, parameter gamma and optional userdata.
            'prec_solvefn':
                Values: function of class IDA_PrecSolveFunction
                Description:
                    Defines a function that solves the preconditioning problem
                    P*z = r where P may be a left or right preconditioner
                    matrix. This function takes as input arguments current time
                    t, current value of y, right-hand side r, result vector z,
                    parameters gamma and delta, input flag lr that determines
                    the flavour of the preconditioner (left = 1, right = 2) and
                    optional userdata.
            'jac_times_vecfn':
                Values: function of class IDA_JacTimesVecFunction
                Description:
                    Defines a function that solves the product of vector v
                    with an (approximate) Jacobian of the system J.
                    This function takes as input arguments:
                        tt is the current value of the independent variable.
                        yy is the current value of the dependent variable vector, y(t).
                        yp is the current value of ˙y(t).
                        rr is the current value of the residual vector F(t, y, y˙).
                        v is the vector by which the Jacobian must be multiplied to 
                            the right.
                        Jv is the computed output vector.
                        cj is the scalar in the system Jacobian, proportional to the 
                            inverse of the step size.
                        user data is a pointer to user data (optional)
            'jac_times_setupfn':
                Values: function of class IDA_JacTimesSetupFunction
                Description:
                    Optional. Default is to internal finite difference with no
                    extra setup.
                    Defines a function that preprocesses and/or evaluates 
                    Jacobian-related data needed by the Jacobiantimes-vector routine
                    This function takes as input arguments:
                        tt is the current value of the independent variable.
                        yy is the current value of the dependent variable vector, y(t).
                        yp is the current value of ˙y(t).
                        rr is the current value of the residual vector F(t, y, y˙).
                        cj is the scalar in the system Jacobian, proportional to the 
                            inverse of the step size.
                        user data is a pointer to user data (optional)
            'err_handler':
                Values: function of class IDA_ErrHandler, default = None
                Description:
                    Defines a function which controls output from the IDA
                    solver
            'err_user_data':
                Description:
                    User data used by 'err_handler', defaults to 'user_data'
            'old_api':
                Values: True (default), False
                Description:
                    Forces use of old api (tuple of 7) if True or
                    new api (namedtuple) if False.
                    Other options may require new api, hence using this should
                    be avoided if possible.
            'onroot':
                Description:
                    If the solver returns ROOT, call this function. If it
                    returns 0, continue solving, otherwise stop.
                    If not given, solver stops after a ROOT
            'ontstop':
                Description:
                    If the solver returns TSTOP, call this function. If it
                    returns 0, continue solving, otherwise stop.
                    If not given, solver stops after a TSTOP
            'validate_flags':
                Values: True, False (=default)
                Description:
                    Controls whether to validate flags as a result of calling
                    `solve`. See the `validate_flags` function for how this
                    affects `solve`.
        """

        super(IDAS, self).set_options( **options)

    cpdef _set_runtime_changeable_options(self, object options,
                                          bint supress_supported_check=False):
        """
          (Re)Set options that can change after IDA_MEM object was allocated -
          mostly useful for values that need to be changed at run-time.
        """
        
        
        super(IDAS, self)._set_runtime_changeable_options(options,
                                              supress_supported_check)

    def init_step(self, DTYPE_t t0, object y0, object yp0,
                   np.ndarray y_ic0_retn = None,
                   np.ndarray yp_ic0_retn = None):
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

        Return values:
         if old_api:
            flag  - status of the computation (successful or error occured)
            t_out - time, where the solver stopped (when no error occured, t_out == t)

         if old_api False (cvode solver):
            A named tuple, with entries:
                flag   = An integer flag (StatusEnumXXX)
                values = Named tuple with entries t and y and ydot. y will
                            correspond to y_ic0_retn value and ydot to yp_ic0_retn!
                errors = Named tuple with entries t and y and ydot
                roots  = Named tuple with entries t and y and ydot
                tstop  = Named tuple with entries t and y and ydot
                message= String with message in case of an error
        """
        soln = super(IDAS, self).init_step(t0, y0, yp0, y_ic0_retn,
                    yp_ic0_retn)
        return soln

    def solve(self, object tspan, object y0,  object yp0):
        """
        Runs the solver.

        Input:
            tspan - an numpy array of times at which the computed value will be
                    returned.  Must contain the start time as first entry.
            y0    - numpy array of initial values
            yp0   - numpy array of initial values of derivatives

        Return values:
         if old_api
            flag   - indicating return status of the solver
            t      - numpy array of times at which the computations were
                     successful
            y      - numpy array of values corresponding to times t
                     (values of y[i, :] ~ t[i])
            yp     - numpy array of derivatives corresponding to times t
                     (values of yp[i, :] ~ t[i])
            t_err  - float or None - if recoverable error occured (for example,
                     reached the maximum number of allowed iterations); this is
                     the time at which it happened
            y_err  - numpy array of values corresponding to time t_err
            yp_err - numpy array of derivatives corresponding to time t_err

         if old_api False (cvode solver):
            A named tuple, with entries:
                flag   = An integer flag (StatusEnum)
                values = Named tuple with entries array t and array y and array ydot
                errors = Named tuple with entries t and y and ydot of error
                roots  = Named tuple with entries array t and array y and array ydot
                tstop  = Named tuple with entries array t and array y and array ydot
                message= String with message in case of an error
        """

        soln = super(IDAS, self).solve(tspan, y0,  yp0)
        return soln

    def step(self, DTYPE_t t, np.ndarray[DTYPE_t, ndim=1] y_retn = None,
                              np.ndarray[DTYPE_t, ndim=1] yp_retn = None):
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
         if old_api:
            flag  - status of the computation (successful or error occured)
            t_out - time, where the solver stopped (when no error occured, t_out == t)

         if old_api False (ida solver):
            A named tuple, with entries:
                flag   = An integer flag (StatusEnumXXX)
                values = Named tuple with entries t and y and ydot. y will
                            correspond to y_retn value and ydot to yp_retn!
                errors = Named tuple with entries t and y and ydot
                roots  = Named tuple with entries t and y and ydot
                tstop  = Named tuple with entries t and y and ydot
                message= String with message in case of an error
        """
        soln = super(IDAS, self).step(t, y_retn, yp_retn)
        return soln

