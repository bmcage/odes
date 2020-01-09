# cython: embedsignature=True
from cpython.exc cimport PyErr_CheckSignals
from collections import namedtuple
from enum import IntEnum
import inspect
from warnings import warn

include "sundials_config.pxi"

import numpy as np
cimport numpy as np

from . import (
    CVODESolveFailed, CVODESolveFoundRoot, CVODESolveReachedTSTOP,
    _get_num_args,
)

from .c_sundials cimport realtype, N_Vector
from .c_nvector_serial cimport *
from .c_sunmatrix cimport *
from .c_sunlinsol cimport *
from .c_sunnonlinsol cimport *

from .cvode cimport (CV_RhsFunction, CV_WrapRhsFunction, CV_RootFunction,
                       CV_WrapRootFunction, CV_JacRhsFunction,
                       CV_WrapJacRhsFunction, CV_PrecSetupFunction,
                       CV_WrapPrecSetupFunction,
                       CV_PrecSolveFunction, CV_WrapPrecSolveFunction,
                       CV_JacTimesVecFunction, CV_WrapJacTimesVecFunction,
                       CV_JacTimesSetupFunction, CV_WrapJacTimesSetupFunction,
                       CV_ContinuationFunction, CV_ErrHandler, 
                       CV_WrapErrHandler, CV_data, CVODE)
from .c_cvodes cimport *
from .common_defs cimport (
    nv_s2ndarray, ndarray2nv_s, ndarray2SUNMatrix, DTYPE_t, INDEX_TYPE_t,
)
from .common_defs import DTYPE, INDEX_TYPE
# this is needed because we want DTYPE and INDEX_TYPE to be
# accessible from python (not only in cython)


# TODO: parallel implementation: N_VectorParallel
# TODO: linsolvers: check the output value for errors
# TODO: optimize code for compiler

SolverReturn = namedtuple(
    "SolverReturn", [
        "flag", "values", "errors", "roots",
        "tstop", "message"
    ]
)

SolverVariables = namedtuple("SolverVariables", ["t", "y"])

class StatusEnum(IntEnum):
    SUCCESS           = CV_SUCCESS              # 0
    TSTOP_RETURN      = CV_TSTOP_RETURN         # 1
    ROOT_RETURN       = CV_ROOT_RETURN          # 2
    WARNING           = CV_WARNING              # 99
    TOO_MUCH_WORK     = CV_TOO_MUCH_WORK        # -1
    TOO_MUCH_ACC      = CV_TOO_MUCH_ACC         # -2
    ERR_FAILURE       = CV_ERR_FAILURE          # -3
    CONV_FAILURE      = CV_CONV_FAILURE         # -4
    LINIT_FAIL        = CV_LINIT_FAIL           # -5
    LSETUP_FAIL       = CV_LSETUP_FAIL          # -6
    LSOLVE_FAIL       = CV_LSOLVE_FAIL          # -7
    RHSFUNC_FAIL      = CV_RHSFUNC_FAIL         # -8
    FIRST_RHSFUNC_ERR = CV_FIRST_RHSFUNC_ERR    # -9
    REPTD_RHSFUNC_ERR = CV_REPTD_RHSFUNC_ERR    # -10
    UNREC_RHSFUNC_ERR = CV_UNREC_RHSFUNC_ERR    # -11
    RTFUNC_FAIL       = CV_RTFUNC_FAIL          # -12
    NLS_INIT_FAIL     = CV_NLS_INIT_FAIL        # -13
    NLS_SETUP_FAIL    = CV_NLS_SETUP_FAIL       # -14
    CONSTR_FAIL       = CV_CONSTR_FAIL          # -15
    NLS_FAIL          = CV_NLS_FAIL             # -16
    MEM_FAIL          = CV_MEM_FAIL             # -20
    MEM_NULL          = CV_MEM_NULL             # -21
    ILL_INPUT         = CV_ILL_INPUT            # -22
    NO_MALLOC         = CV_NO_MALLOC            # -23
    BAD_K             = CV_BAD_K                # -24
    BAD_T             = CV_BAD_T                # -25
    BAD_DKY           = CV_BAD_DKY              # -26
    TOO_CLOSE         = CV_TOO_CLOSE            # -27
    VECTOROP_ERR      = CV_VECTOROP_ERR         # -28
    NO_QUAD           = CV_NO_QUAD              # -30
    QRHSFUNC_FAIL     = CV_QRHSFUNC_FAIL        # -31
    FIRST_QRHSFUNC_ERR= CV_FIRST_QRHSFUNC_ERR   # -32
    REPTD_QRHSFUNC_ERR= CV_REPTD_QRHSFUNC_ERR   # -33
    UNREC_QRHSFUNC_ERR= CV_UNREC_QRHSFUNC_ERR   # -34
    NO_SENS           = CV_NO_SENS              # -40
    SRHSFUNC_FAIL     = CV_SRHSFUNC_FAIL        # -41
    FIRST_SRHSFUNC_ERR= CV_FIRST_SRHSFUNC_ERR   # -42
    REPTD_SRHSFUNC_ERR= CV_REPTD_SRHSFUNC_ERR   # -43
    UNREC_SRHSFUNC_ERR= CV_UNREC_SRHSFUNC_ERR   # -44
    BAD_IS            = CV_BAD_IS               # -45
    NO_QUADSENS       = CV_NO_QUADSENS          # -50
    QSRHSFUNC_FAIL    = CV_QSRHSFUNC_FAIL       # -51
    FIRST_QSRHSFUNC_ERR=CV_FIRST_QSRHSFUNC_ERR  # -52
    REPTD_QSRHSFUNC_ERR=CV_REPTD_QSRHSFUNC_ERR  # -53
    UNREC_QSRHSFUNC_ERR=CV_UNREC_QSRHSFUNC_ERR  # -54
    UNRECOGNIZED_ERR  = CV_UNRECOGNIZED_ERR     # -99

STATUS_MESSAGE = {
    StatusEnum.SUCCESS: "Successful function return.",
    StatusEnum.TSTOP_RETURN: "Reached specified stopping point",
    StatusEnum.ROOT_RETURN: "Found one or more roots",
    StatusEnum.WARNING: "Succeeded but something unusual happened",
    StatusEnum.TOO_MUCH_WORK: "Could not reach endpoint",
    StatusEnum.TOO_MUCH_ACC: "Could not satisfy accuracy",
    StatusEnum.ERR_FAILURE: "Error test failures occurred too many times during one internal time step or minimum step size was reached.",
    StatusEnum.CONV_FAILURE: "Convergence test failures occurred too many times during one internal time step or minimum step size was reached.",
    StatusEnum.LINIT_FAIL: "The linear solver’s initialization function failed.",
    StatusEnum.LSETUP_FAIL: "The linear solver’s setup function failed in an unrecoverable manner.",
    StatusEnum.LSOLVE_FAIL: "The linear solver’s solve function failed in an unrecoverable manner.",
    StatusEnum.RHSFUNC_FAIL: "The right-hand side function failed in an unrecoverable manner.",
    StatusEnum.FIRST_RHSFUNC_ERR: "The right-hand side function failed at the first call.",
    StatusEnum.REPTD_RHSFUNC_ERR: "The right-hand side function had repeated recoverable errors.",
    StatusEnum.UNREC_RHSFUNC_ERR: "The right-hand side function had a recoverable error, but no recovery is possible.",
    StatusEnum.RTFUNC_FAIL: "The rootfinding function failed in an unrecoverable manner.",
    StatusEnum.NLS_INIT_FAIL: "The nonlinear solver's init routine failed.",
    StatusEnum.NLS_SETUP_FAIL: "The nonlinear solver setup failed unrecoverably.",
    StatusEnum.CONSTR_FAIL: "The inequality constraints could not be met",
    StatusEnum.NLS_FAIL: "The nonlinear solver failed in an unrecoverable manner",
    StatusEnum.MEM_FAIL: "A memory allocation failed.",
    StatusEnum.MEM_NULL: "The cvode_mem argument was NULL.",
    StatusEnum.ILL_INPUT: "One of the function inputs is illegal.",
    StatusEnum.NO_MALLOC: "The cvode memory block was not allocated by a call to CVodeMalloc.",
    StatusEnum.BAD_K: "The derivative order k is larger than the order used.",
    StatusEnum.BAD_T: "The time t is outside the last step taken.",
    StatusEnum.BAD_DKY: "The output derivative vector is NULL.",
    StatusEnum.TOO_CLOSE: "The output and initial times are too close to each other.",
    StatusEnum.VECTOROP_ERR: "Vector operation error",
    StatusEnum.NO_QUAD: "Quadrature integration not activated.",
    StatusEnum.QRHSFUNC_FAIL: "The quadrature right-hand side routine failed in an unrecoverable manner.",
    StatusEnum.FIRST_QRHSFUNC_ERR: "The quadrature right-hand side routine failed at the first call.",
    StatusEnum.REPTD_QRHSFUNC_ERR: "Repeated recoverable quadrature right-hand side function errors.",
    StatusEnum.UNREC_QRHSFUNC_ERR: "The quadrature right-hand side failed in a recoverable manner, but no recovery is possible.",
    StatusEnum.NO_SENS: "Forward sensitivity analysis not activated.",
    StatusEnum.SRHSFUNC_FAIL: "The quadrature sensitivity right-hand side routine failed in an unrecoverable manner.",
    StatusEnum.FIRST_SRHSFUNC_ERR: "The sensitivity right-hand side routine failed at the first call.",
    StatusEnum.REPTD_SRHSFUNC_ERR: "Repeated recoverable sensitivity right-hand side function errors.",
    StatusEnum.UNREC_SRHSFUNC_ERR: "The sensitivity right-hand side failed in a recoverable manner, but no recovery is possible.",
    StatusEnum.BAD_IS: "Illegal value for is parameter.",
    StatusEnum.NO_QUADSENS: "Forward sensitivity analysis for quadrature variables not activated.",
    StatusEnum.QSRHSFUNC_FAIL: "The quadrature sensitivity right-hand side routine failed in an unrecoverable manner.",
    StatusEnum.FIRST_QSRHSFUNC_ERR: "The quadrature sensitivity right-hand side routine failed at the first call.",
    StatusEnum.REPTD_QSRHSFUNC_ERR: "Repeated recoverable quadrature sensitivity right-hand side function errors.",
    StatusEnum.UNREC_QSRHSFUNC_ERR: "The quadrature sensitivity right-hand side failed in a recoverable manner, but no recovery is possible.",
    StatusEnum.UNRECOGNIZED_ERR: "Unrecognized Error",
}

WARNING_STR = "Solver succeeded with flag {} and finished at {} with values {}"


# Auxiliary data carrying runtime vales for the CVODE solver
cdef class CVS_data(CV_data):
    def __cinit__(self, N, Ns=0):
        super(CVS_data, self).__init__(N)
        
        self.yS_tmp = np.empty(Ns, DTYPE)
        self.ySdot_tmp = np.empty(Ns, DTYPE)

cdef class CVODES(CVODE):

    def __cinit__(self, Rfn, **options):
        """
        Initialize the CVODE Solver and it's default values

        Input:
            Rfn     - right-hand-side function
            options - additional options for initialization, for the list
                      of supported options and their values see set_options()

        """
        super(CVODES, self).__init__(Rfn, **options)
        
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
                    Deprecated, does nothing
                Note: Changeable at runtime.
            'implementation':
                Values: 'serial' (= default), 'parallel'
                Description:
                    Using serial or parallel implementation of the solver.
                    #TODO: curently only serial implementation is working
            'lmm_type'  - linear multistep method
                Values: 'ADAMS', 'BDF' (= default)
                Description:
                    Recommended combination:

                    Problem  | 'lmm_type' | nonlinearsolver module
                    ------------------------------------
                    nonstiff | 'ADAMS'    | 'fixedpoint' (SUNNonlinearSolver_FixedPoint )
                    stiff    | 'BDF'      | 'newton' (SUNNonlinearSolver_Newton)

                    See also 'iter_type'.
            'nonlinsolver' - nonlinear solver iteration
                Values: 'newton' (= default), 'fixedpoint'
                Description:
                    See 'lmm_type'.

            'rfn':
                Values: function of class CV_RhsFunction or a python function
                        with signature (t, y, yp) or (t, y, yp, userdata)
                Description:
                    Defines the right-hand-side function (which has to be
                    a subclass of CV_RhsFunction class, or a normal python
                    function with signature (t, y, yp) or (t, y, yp, userdata)).
                    This function takes as input arguments current time t,
                    current value of y, and yp must be used as output numpy
                    array of returned values of \dot{y} at t.
                    Optional userdata. Return value is 0 if successfull.
                    This option is mandatory.
            'rootfn':
                Values: function of class CV_RootFunction or a python function
                    with signature (t, y, g, user_data)
                Description:
                    Defines a function that fills a vector 'g' with values
                    (g is a vector of size k). Vector 'g represents conditions
                    which when satisfied, the solver stops. Namely, the solver
                    stops when condition g[i] is zero for some i.
                    For 'user_data' argument see 'user_data' option
                    above.
                Note: Changeable at runtime.
            'nr_rootfns':
                Value: integer
                Description:
                    The length of the array returned by 'rootfn' (see above),
                    i.e. number of conditions for which we search the value 0.
                Note: Changeable at runtime.
            'jacfn':
                Values: function of class CV_JacRhsFunction
                Description:
                    Defines the jacobian function and has to be a subclass
                    of CV_JacRhsFunction class or python function. This function
                    takes as input arguments current time t, current value of y,
                    current value of f(t,y), and 
                    a 2D numpy array of returned jacobian and optional userdata.
                    Return value is 0 if successfull.
                    Jacobian is only used for dense or lapackdense linear solver
                TODO: cvode supports Jacobian for band also, this is not supported.
            'rtol':
                Values: float,  1e-6 = default
                Description:
                    Relative tolerancy.
                Note: Changeable at runtime.
            'atol':
                Values: float or numpy array of floats,  1e-12 = default
                Description:
                    Absolute tolerancy
                Note: Changeable at runtime.
            'order':
                Values: 1, 2, 3, 4, 5 (= default)
                Description:
                    Specifies the maximum order of the linear multistep method.
            'max_steps':
                Values: positive integer, 0 = default (uses value of 500)
                Description:
                    Maximum number of (internally defined) steps allowed
                    during one call to the solver.
            'max_step_size':
                Values: non-negative float, 0.0 = default
                Description:
                    Restricts the maximal (absolute) internal step value taken
                    by the solver. The value of 0.0 uses
                    no restriction on the maximal step size.
            'min_step_size':
                Values: non-negative float, 0.0 = default
                Description:
                    Restricts the minimal (absolute) internal step value taken
                    by the solver. The value of 0.0 uses
                    no restriction.
            'first_step_size':
                Values: float , 0.0 = default
                Description:
                    Sets the first step size. DAE solver can suffer on the
                    first step, so set this to circumvent this. The value
                    of 0.0 uses the solver's internal default value.
            'linsolver':
                Values: 'dense' (= default), 'lapackdense', 'band',
                        'lapackband', 'spgmr', 'spbcgs', 'sptfqmr'
                Description:
                    Specifies used linear solver.
                    Limitations: Linear solvers for dense and band matrices
                                 can be used only for serial implementation.
                                 For parallel implementation use_relaxation
                                 use lapackdense or lapackband respectively.
                    TODO: to add new solvers: pcg, spfgmr, superlumt, klu
                    
            'nonlinsolver':
                Values: 'newton' (= default), 'fixedpoint'
                Description:
                    Specifies the used nonlinear solver.
                    
            'lband', 'uband':
                Values: non-negative integer, 0 = default
                Description:
                    Specifies the width of the lower band (below the main
                    diagonal) and/or upper diagonal (above the main diagonal).
                    So the total band width is lband + 1 + uband, where the 1
                    stands for the main diagonal
                    (specially, lband = uband = 0 [that is the default]
                    denotes the band width = 1, i.e. the main diagonal only).
                    Used only if 'linsolver' is band.
            'maxl':
                Values: 0 (= default), 1, 2, 3, 4, 5
                Description:
                    Dimension of the number of used Krylov subspaces
                    (used only by 'spgmr', 'spbcgs', 'sptfqmr' linsolvers)
            'tstop':
                Values: float, 0.0 = default
                Description:
                    Maximum time until which we perform the computations.
                    Default is 0.0. Once the value is set 0.0, it cannot be
                    disabled (but it will be automatically disabled when tstop
                    is reached and has to be reset again in next run).
                Note: Changeable at runtime.
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
                    residual function 'Rfn' (see below) and Jacobi function
                    'jacfn' (if specified one).
            'precond_type':
                default = None
            'prec_setupfn':
                Values: function of class CV_PrecSetupFunction
                Description:
                    Defines a function that setups the preconditioner on change
                    of the Jacobian. This function takes as input arguments
                    current time t, current value of y, flag jok that indicates
                    whether Jacobian-related data has to be updated, flag jcurPtr
                    that should be set to True (jcurPtr.value = True) if Jacobian
                    data was recomputed, parameter gamma and optional userdata.
            'prec_solvefn':
                Values: function of class CV_PrecSolveFunction
                Description:
                    Defines a function that solves the preconditioning problem
                    P*z = r where P may be a left or right preconditioner
                    matrix. This function takes as input arguments current time
                    t, current value of y, right-hand side r, result vector z,
                    parameters gamma and delta, input flag lr that determines
                    the flavour of the preconditioner (left = 1, right = 2) and
                    optional userdata.
            'jac_times_vecfn':
                Values: function of class CV_JacTimesVecFunction
                Description:
                    Defines a function that solves the product of vector v
                    with an (approximate) Jacobian of the system J.
                    This function takes as input arguments the vector v,
                    result vector Jv, current time t, current value of y,
                    and optional userdata.
            'jac_times_setupfn':
                Values: function of class CV_JacTimesSetupFunction
                Description:
                    Optional. Default is to internal finite difference with no
                    extra setup.
            'bdf_stability_detection':
                default = False, only used if lmm_type == 'bdf
            'max_conv_fails':
                default = 0,
            'max_nonlin_iters':
                default = 0,
            'nonlin_conv_coef':
                default = 0,
            'err_handler':
                Values: function of class CV_ErrHandler, default = None
                Description:
                    Defines a function which controls output from the CVODE
                    solver
            'err_user_data':
                Description:
                    User data used by 'err_handler', defaults to 'user_data'
            'old_api':
                Values: True (default), False
                Description:
                    Forces use of old api (tuple of 5, with last two items
                    overloaded) if True or new api (namedtuple) if False.
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

        # Update values of all supplied options
        super(CVODES, self).set_options(**options)

    cpdef _set_runtime_changeable_options(self, object options,
                                          bint supress_supported_check=False):
        """
          (Re)Set options that can change after CV_MEM object was allocated -
          mostly useful for values that need to be changed at run-time.
        """
        
        super(CVODES, self)._set_runtime_changeable_options(options,
                                              supress_supported_check)

    def init_step(self, DTYPE_t t0, object y0):
        """
        Initialize the solver and all the internal variables. This assumes
        the call to 'set_options()' to be done and hence all the information
        for correct solver initialization to be available.
        An error is raised for almost all errors.

        Return values:
         if old_api:
            flag  - boolean status of the computation (successful or error occured)
            t_out - initial time

         if old_api False (cvode solver):
            A named tuple, with entries:
                flag   = An integer flag (StatusEnumXXX)
                values = Named tuple with entries t and y and ydot. y will
                            correspond to y_retn value and ydot to yp_retn!
                errors = Named tuple with entries t_err and y_err
                roots  = Named tuple with entries t_roots and y_roots
                tstop  = Named tuple with entries t_stop and y_tstop
                message= String with message in case of an error

        Note: some options can be re-set also at runtime. See 'reinit_IC()'
        """
        
        soln = super(CVODES, self).init_step(t0, y0)
        return soln

    def solve(self, object tspan, object y0):
        """
        Runs the solver.

        Input:
            tspan - an list/array of times at which the computed value will be
                    returned. Must contain the start time as first entry..
            y0    - list/numpy array of initial values

        Return values:
         if old_api
            flag   - indicating return status of the solver
            t      - numpy array of times at which the computations were successful
            y      - numpy array of values corresponding to times t (values of y[i, :] ~ t[i])
            t_err  - float or None - if recoverable error occured (for example reached maximum
                     number of allowed iterations), this is the time at which it happened
            y_err  - numpy array of values corresponding to time t_err
         if old_api False (cvode solver):
            A named tuple, with entries:
                flag   = An integer flag (StatusEnum)
                values = Named tuple with entries t and y
                errors = Named tuple with entries t and y
                roots  = Named tuple with entries t and y
                tstop  = Named tuple with entries t and y
                message= String with message in case of an error
        """
        soln = super(CVODES, self).solve(tspan, y0)
        return soln

    def step(self, DTYPE_t t, np.ndarray[DTYPE_t, ndim=1] y_retn = None):
        """
        Method for calling successive next step of the CVODE solver to allow
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
            t_out - time, where the solver stopped
                    (when no error occured, t_out == t)
        """
        soln = super(CVODES, self).step(t, y_retn)
        return soln

    def get_info(self):
        """
        Returns a dict with information about the solver, like number
        of calls to the user's right-hand side function.

        """
        info = super(CVODES, self).get_info()

        return info
