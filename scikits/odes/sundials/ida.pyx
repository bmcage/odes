from cpython.exc cimport PyErr_CheckSignals
from collections import namedtuple
from enum import IntEnum
import inspect
from warnings import warn

import numpy as np
cimport numpy as np

from .c_sundials cimport realtype, N_Vector
from .c_ida cimport *
from .common_defs cimport (nv_s2ndarray, ndarray2nv_s, ndarray2DlsMatd)
from . import IDASolveFailed, IDASolveFoundRoot, IDASolveReachedTSTOP

# TODO: parallel implementation: N_VectorParallel
# TODO: linsolvers: check the output value for errors
# TODO: unify using float/double/realtype variable
# TODO: optimize code for compiler

SolverReturn = namedtuple(
    "SolverReturn", [
        "flag", "values", "errors", "roots",
        "tstop", "message"
    ]
)

SolverVariables = namedtuple("SolverVariables", ["t", "y", "ydot"])

class StatusEnumIDA(IntEnum):
    SUCCESS = IDA_SUCCESS
    TSTOP_RETURN = IDA_TSTOP_RETURN
    ROOT_RETURN = IDA_ROOT_RETURN

    WARNING = IDA_WARNING

    TOO_MUCH_WORK = IDA_TOO_MUCH_WORK
    TOO_MUCH_ACC = IDA_TOO_MUCH_ACC
    ERR_FAIL = IDA_ERR_FAIL
    CONV_FAIL = IDA_CONV_FAIL

    LINIT_FAIL = IDA_LINIT_FAIL
    LSETUP_FAIL = IDA_LSETUP_FAIL
    LSOLVE_FAIL = IDA_LSOLVE_FAIL
    RES_FAIL = IDA_RES_FAIL
    REP_RES_ERR = IDA_REP_RES_ERR
    RTFUNC_FAIL = IDA_RTFUNC_FAIL
    CONSTR_FAIL= IDA_CONSTR_FAIL

    FIRST_RES_FAIL = IDA_FIRST_RES_FAIL
    LINESEARCH_FAIL = IDA_LINESEARCH_FAIL
    NO_RECOVERY = IDA_NO_RECOVERY

    MEM_NULL= IDA_MEM_NULL
    MEM_FAIL = IDA_MEM_FAIL
    ILL_INPUT = IDA_ILL_INPUT
    NO_MALLOC = IDA_NO_MALLOC
    BAD_EWT = IDA_BAD_EWT
    BAD_K = IDA_BAD_K
    BAD_T = IDA_BAD_T
    BAD_DKY = IDA_BAD_DKY

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
    StatusEnumIDA.MEM_NULL: "Integrator memory is NULL.",
    StatusEnumIDA.MEM_FAIL: "A memory request failed.",
    StatusEnumIDA.ILL_INPUT: "Invalid input detected.",
    StatusEnumIDA.NO_MALLOC: "Attempt to call before IDAMalloc. Allocate memory first.",
    StatusEnumIDA.BAD_EWT: "Some initial ewt component = 0.0 illegal.",
    StatusEnumIDA.BAD_K: "Illegal value for k. If the requested k is not in the range 0,1,...,order used ",
    StatusEnumIDA.BAD_T: "Illegal value for t. If t is not within the interval of the last step taken.",
    StatusEnumIDA.BAD_DKY: "The dky vector is NULL",

}

WARNING_STR = "Solver succeeded with flag {} and finished at {} with values {}"

# Right-hand side function
cdef class IDA_RhsFunction:
    """
    Prototype for rhs function.

    Note that evaluate must return a integer, 0 for success, positive for
    recoverable failure, negative for unrecoverable failure (as per IDA
    documentation).
    """
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       np.ndarray[DTYPE_t, ndim=1] result,
                       object userdata = None) except? -1:
        return 0

cdef class IDA_WrapRhsFunction(IDA_RhsFunction):
    cpdef set_resfn(self, object resfn):
        """
        set some residual equations as a ResFunction executable class
        """
        self.with_userdata = 0
        nrarg = len(inspect.getargspec(resfn)[0])
        if nrarg > 5:
            # hopefully a class method
            self.with_userdata = 1
        elif nrarg == 5 and inspect.isfunction(resfn):
            self.with_userdata = 1
        self._resfn = resfn

    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       np.ndarray[DTYPE_t, ndim=1] result,
                       object userdata = None) except? -1:
        if self.with_userdata == 1:
            user_flag = self._resfn(t, y, ydot, result, userdata)
        else:
            user_flag = self._resfn(t, y, ydot, result)
        if user_flag is None:
            user_flag = 0
        return user_flag


cdef int _res(realtype tt, N_Vector yy, N_Vector yp,
              N_Vector rr, void *auxiliary_data):
    """ function with the signature of IDAResFn """
    cdef np.ndarray[DTYPE_t, ndim=1] residual_tmp, yy_tmp, yp_tmp

    aux_data = <IDA_data> auxiliary_data
    cdef bint parallel_implementation = aux_data.parallel_implementation

    if parallel_implementation:
        raise NotImplemented
    else:
        yy_tmp = aux_data.yy_tmp
        yp_tmp = aux_data.yp_tmp
        residual_tmp = aux_data.residual_tmp

        nv_s2ndarray(yy, yy_tmp)
        nv_s2ndarray(yp, yp_tmp)

    user_flag = aux_data.res.evaluate(tt, yy_tmp, yp_tmp, residual_tmp, aux_data.user_data)

    if parallel_implementation:
        raise NotImplemented
    else:
        ndarray2nv_s(rr, residual_tmp)

    return user_flag

# Root function
cdef class IDA_RootFunction:
    """
    Prototype for root function.

    Note that evaluate must return a integer, 0 for success, non-zero for error
    (as per IDA documentation).
    """
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       np.ndarray[DTYPE_t, ndim=1] g,
                       object userdata = None) except? -1:
        return 0

cdef class IDA_WrapRootFunction(IDA_RootFunction):
    cpdef set_rootfn(self, object rootfn):
        """
        set root-ing condition(equations) as a RootFunction executable class
        """
        self.with_userdata = 0
        nrarg = len(inspect.getargspec(rootfn)[0])
        if nrarg > 5:
            #hopefully a class method, self gives 5 arg!
            self.with_userdata = 1
        elif nrarg == 5 and inspect.isfunction(rootfn):
            self.with_userdata = 1
        self._rootfn = rootfn

    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       np.ndarray[DTYPE_t, ndim=1] g,
                       object userdata = None) except? -1:
        if self.with_userdata == 1:
            user_flag = self._rootfn(t, y, ydot, g, userdata)
        else:
            user_flag = self._rootfn(t, y, ydot, g)
        if user_flag is None:
            user_flag = 0
        return user_flag

cdef int _rootfn(realtype t, N_Vector yy, N_Vector yp,
                 realtype *gout, void *auxiliary_data):
    """ function with the signature of IDARootFn """

    aux_data = <IDA_data> auxiliary_data
    cdef bint parallel_implementation = aux_data.parallel_implementation

    if parallel_implementation:
        raise NotImplemented
    else:
        yy_tmp = aux_data.yy_tmp
        yp_tmp = aux_data.yp_tmp
        g_tmp  = aux_data.g_tmp

        nv_s2ndarray(yy, yy_tmp)
        nv_s2ndarray(yp, yp_tmp)

    user_flag = aux_data.rootfn.evaluate(t, yy_tmp, yp_tmp, g_tmp, aux_data.user_data)

    cdef int i
    if parallel_implementation:
        raise NotImplemented
    else:
        for i in np.arange(np.alen(g_tmp)):
            gout[i] = <realtype> g_tmp[i]

    return user_flag

# Jacobian function
cdef class IDA_JacRhsFunction:
    """
    Prototype for jacobian function.

    Note that evaluate must return a integer, 0 for success, positive for
    recoverable failure, negative for unrecoverable failure (as per IDA
    documentation).
    """
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       DTYPE_t cj,
                       np.ndarray J) except? -1:
        """
        Returns the Jacobi matrix of the residual function, as
            d(res)/d y + cj d(res)/d ydot
        (for dense the full matrix, for band only bands). Result has to be
        stored in the variable J, which is preallocated to the corresponding
        size.

        This is a generic class, you should subclass is for the problem specific
        purposes."
        """
        return 0

cdef class IDA_WrapJacRhsFunction(IDA_JacRhsFunction):
    cpdef set_jacfn(self, object jacfn):
        """
        Set some jacobian equations as a JacResFunction executable class.
        """
        self._jacfn = jacfn

    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       DTYPE_t cj,
                       np.ndarray J) except? -1:
        """
        Returns the Jacobi matrix (for dense the full matrix, for band only
        bands. Result has to be stored in the variable J, which is preallocated
        to the corresponding size.
        """
##        if self.with_userdata == 1:
##            self._jacfn(t, y, ydot, cj, J, userdata)
##        else:
##            self._jacfn(t, y, ydot, cj, J)
        user_flag = self._jacfn(t, y, ydot, cj, J)
        if user_flag is None:
            user_flag = 0
        return user_flag

cdef int _jacdense(long int Neq, realtype tt, realtype cj,
            N_Vector yy, N_Vector yp, N_Vector rr, DlsMat Jac,
            void *auxiliary_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3):
    """function with the signature of IDADlsDenseJacFn """
    cdef np.ndarray[DTYPE_t, ndim=1] yy_tmp, yp_tmp
    cdef np.ndarray jac_tmp

    aux_data = <IDA_data> auxiliary_data
    cdef bint parallel_implementation = aux_data.parallel_implementation
    if parallel_implementation:
        raise NotImplemented
    else:
        yy_tmp = aux_data.yy_tmp
        yp_tmp = aux_data.yp_tmp
        if aux_data.jac_tmp is None:
            N = np.alen(yy_tmp)
            aux_data.jac_tmp = np.empty((N,N), float)
        jac_tmp = aux_data.jac_tmp

        nv_s2ndarray(yy, yy_tmp)
        nv_s2ndarray(yp, yp_tmp)
    user_flag = aux_data.jac.evaluate(tt, yy_tmp, yp_tmp, cj, jac_tmp)

    if parallel_implementation:
        raise NotImplemented
    else:
        #we convert the python jac_tmp array to DslMat of sundials
        ndarray2DlsMatd(Jac, jac_tmp)

    return user_flag

cdef class IDA_ContinuationFunction:
    """
    Simple wrapper for functions called when ROOT or TSTOP are returned.
    """
    def __cinit__(self, fn):
        self._fn = fn
    cpdef int evaluate(self,
                       DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] yp,
                       IDA solver):
        return self._fn(t, y, yp, solver)

def no_continue_fn(t, y, yp, solver):
    return 1

cdef class IDA_ErrHandler:
    cpdef evaluate(self,
                   int error_code,
                   bytes module,
                   bytes function,
                   bytes msg,
                   object user_data = None):
        """ format that error handling functions must match """
        pass

cdef class IDA_WrapErrHandler(IDA_ErrHandler):
    cpdef set_err_handler(self, object err_handler):
        """
        set some (c/p)ython function as the error handler
        """
        nrarg = len(inspect.getargspec(err_handler)[0])
        self.with_userdata = (nrarg > 5) or (
            nrarg == 5 and inspect.isfunction(err_handler)
        )
        self._err_handler = err_handler

    cpdef evaluate(self,
                   int error_code,
                   bytes module,
                   bytes function,
                   bytes msg,
                   object user_data = None):
        if self.with_userdata == 1:
            self._err_handler(error_code, module, function, msg, user_data)
        else:
            self._err_handler(error_code, module, function, msg)

cdef void _ida_err_handler_fn(
    int error_code, const char *module, const char *function, char *msg,
    void *eh_data
):
    """
    function with the signature of IDAErrHandlerFn, that calls python error
    handler
    """
    aux_data = <IDA_data> eh_data
    aux_data.err_handler.evaluate(error_code,
                                  module,
                                  function,
                                  msg,
                                  aux_data.err_user_data)


cdef class IDA_data:
    def __cinit__(self, N):
        self.parallel_implementation = False
        self.user_data = None
        self.err_user_data = None

        self.yy_tmp = np.empty(N, float)
        self.yp_tmp = np.empty(N, float)
        self.residual_tmp = np.empty(N, float)
        self.jac_tmp = None
        self.g_tmp = None

cdef class IDA:

    def __cinit__(self, Rfn, **options):
        """
        Initialize the IDA Solver and it's default values

        Input:
            Rfn     - residual function
            options - additional options for initialization
        """

        default_values = {
            'implementation': 'serial',
            'rtol': 1e-6, 'atol': 1e-12,
            'linsolver': 'dense',
            'lband': 0,'uband': 0,
            'maxl': 0,
            'tstop': 0.,
            'order': 0,
            'max_step_size': 0.,
            'first_step_size': 0.,
            'max_steps': 0,
            'max_conv_fails': 0,  # sundials default is 10
            'max_nonlin_iters': 0,
            'nonlin_conv_coef': 0.,
            'compute_initcond': None,
            'compute_initcond_t0': 0.01,
            'constraints_idx': None,
            'constraints_type': None,
            'algebraic_vars_idx':None,
            'exclude_algvar_from_error': False,
            'one_step_compute': False,
            'user_data': None,
            'rfn': None,
            'rootfn': None,
            'nr_rootfns': 0,
            'jacfn': None,
            'err_handler': None,
            'err_user_data': None,
            'old_api': None,
            'onroot': None,
            'ontstop': None,
            'validate_flags': None,
            }

        self.verbosity = 1
        self.options = default_values
        self.N       = -1
        self._old_api = True # use old api by default
        self._step_compute = False #avoid dict lookup
        self._validate_flags = False # don't validate by default
        self.set_options(rfn=Rfn, **options)
        self._ida_mem = NULL
        self.initialized = False

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
                    If given problem is of type DAE, some items of the residual
                    vector returned by the 'resfn' have to be treated as
                    algebraic variables. These are denoted by the position
                    (index) in the residual vector.
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

        for (key, value) in options.items():
            if key.lower() in self.options:
                self.options[key.lower()] = value
            else:
                raise ValueError("Option '%s' is not supported by solver" % key)

        self._set_runtime_changeable_options(options)

    cpdef _set_runtime_changeable_options(self, object options,
                                          bint supress_supported_check=False):
        """
          (Re)Set options that can change after IDA_MEM object was allocated -
          mostly useful for values that need to be changed at run-time.
        """
        cdef int flag
        cdef void* ida_mem = self._ida_mem

        if ida_mem is NULL:
            return 0

        # Check whether options are all only supported - this check can
        # be supressed by 'supress_supported_check = True'
        if not supress_supported_check:
            for opt in options.keys():
                if not opt in ['atol', 'rtol', 'tstop', 'rootfn',
                               'nr_rootfns', 'one_step_compute']:
                    raise ValueError("Option '%s' can''t be set runtime." % opt)

        # Verbosity level
        if ('verbosity' in options) and (options['verbosity'] is not None):
            verbosity = options['verbosity']
            self.options['verbosity'] = verbosity
            self.verbosity = verbosity
            warn("verbosity is deprecated, control output via logging and "
                 "err_handler", DeprecationWarning
            )


        # Root function
        if ('rootfn' in options) and (options['rootfn'] is not None):
            # TODO: Unsetting the rootfn?
            rootfn = options['rootfn']
            self.options['rootfn'] = rootfn

            nr_rootfns = options['nr_rootfns']
            self.options['nr_rootfns'] = nr_rootfns
            if nr_rootfns is None:
                raise ValueError('Number of root-ing functions ''nr_rootfns'' '
                                 'must be specified.')
            if not isinstance(rootfn, IDA_RootFunction):
                tmpfun = IDA_WrapRootFunction()
                tmpfun.set_rootfn(rootfn)
                rootfn = tmpfun
                self.options['rootfn'] = tmpfun

            self.aux_data.rootfn = rootfn
            self.aux_data.g_tmp  = np.empty([nr_rootfns,], float)

            flag = IDARootInit(ida_mem, nr_rootfns, _rootfn)

            if flag == IDA_SUCCESS:
                pass
            if flag == IDA_ILL_INPUT:
                raise ValueError('IDARootInit: Function ''rootfn'' is NULL '
                                 'but ''nr_rootfns'' > 0')
            elif flag == IDA_MEM_FAIL:
                raise MemoryError('IDARootInit: Memory allocation error')


        # Set tolerances
        cdef N_Vector atol
        cdef np.ndarray[DTYPE_t, ndim=1] np_atol

        if 'atol' in options:
            opts_atol = options['atol']
            self.options['atol'] = opts_atol
        else:
            opts_atol = None

        if 'rtol' in options:
            opts_rtol = options['rtol']
            self.options['rtol'] = opts_rtol
        else:
            opts_rtol = None

        if not ((opts_rtol is None) and (opts_atol is None)):
            if opts_rtol is None:
                opts_rtol = self.options['rtol']
            else:
                if not np.isscalar(options['rtol']) :
                    raise ValueError("For IDA solver 'rtol' must be a scalar")

            if opts_atol is None:
                opts_atol = self.options['atol']
            else:
                if not (self.atol is NULL):
                    N_VDestroy(self.atol)
                    self.atol = NULL

            if np.isscalar(opts_atol):
                flag = IDASStolerances(ida_mem, <realtype> opts_rtol,
                                                <realtype> opts_atol)
            else:
                np_atol = np.asarray(opts_atol)
                if np.alen(np_atol) != self.N:
                    raise ValueError("Array length inconsistency: 'atol' "
                                     "lenght (%i) differs from problem size "
                                     "(%i)." % (np.alen(np_atol), self.N))

                if self.parallel_implementation:
                    raise NotImplemented
                else:
                    atol = N_VMake_Serial(self.N, <realtype *> np_atol.data)
                    flag = IDASVtolerances(ida_mem, <realtype> opts_rtol, atol)

                    self.atol = atol

            if flag == IDA_ILL_INPUT:
                raise ValueError("IDATolerances: negative 'atol' or 'rtol' value.")
        #TODO: implement IDAFWtolerances(ida_mem, efun)

        # Set tstop
        if ('tstop' in options) and (options['tstop'] is not None):
            opts_tstop = options['tstop']
            self.options['tstop'] = opts_tstop
            if (not opts_tstop is None) and (opts_tstop > 0.):
               flag = IDASetStopTime(ida_mem, <realtype> opts_tstop)
               if flag == IDA_ILL_INPUT:
                   raise ValueError('IDASetStopTime::Stop value is beyond '
                                    'current value.')

        # Force old/new api
        if options.get('old_api') is not None:
            if not options['old_api'] in [True, False]:
                raise ValueError('Option old_api must be True or False')
            self._old_api = options['old_api']

        if options.get('one_step_compute') is not None:
            if not options['one_step_compute'] in [True, False]:
                raise ValueError('Option one_step_compute must be True or False')
            if options['one_step_compute'] and self._old_api:
                raise ValueError("Option 'one_step_compute' requires 'old_api' to be False")
            self.options['one_step_compute'] = options['one_step_compute']
            self._step_compute = self.options['one_step_compute']
        # Set onroot
        if options.get('onroot', None) is not None:
            fn = options['onroot']
            if not isinstance(fn, IDA_ContinuationFunction):
                fn = IDA_ContinuationFunction(fn)
            self.options['onroot'] = fn
        elif self.options.get('onroot', None) is None:
            self.options['onroot'] = IDA_ContinuationFunction(no_continue_fn)

        # Set ontstop
        if options.get('ontstop', None) is not None:
            fn = options['ontstop']
            if not isinstance(fn, IDA_ContinuationFunction):
                fn = IDA_ContinuationFunction(fn)
            self.options['ontstop'] = fn
        elif self.options.get('ontstop', None) is None:
            self.options['ontstop'] = IDA_ContinuationFunction(no_continue_fn)

        # Set validate status
        if options.get('validate_flags') is not None:
            validate_flags = options["validate_flags"]
            if not validate_flags in [True, False]:
                raise ValueError("Option 'validate_flags' must be True or False")
            if validate_flags and self._old_api:
                raise ValueError("Option 'validate_flags' requires 'old_api' to be False")
            self.options['validate_flags'] = validate_flags
            self._validate_flags = options["validate_flags"]

    def init_step(self, double t0, object y0, object yp0,
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
        cdef int flag
        cdef float time
        cdef np.ndarray[DTYPE_t, ndim=1] np_y0
        cdef np.ndarray[DTYPE_t, ndim=1] np_yp0
        np_y0  = np.asarray(y0, dtype=float)
        np_yp0 = np.asarray(yp0, dtype=float)

        flag, time = self._init_step(t0, np_y0, np_yp0, y_ic0_retn, yp_ic0_retn)

        if self._old_api:
            warn("Old api is deprecated, move to new api", DeprecationWarning)

        if not (flag == IDA_SUCCESS):
            if self._old_api:
                print('IDAInitCond: Error occured during computation'
                      ' of initial condition, flag', flag)
                return (False, time)
            else:
                soln = SolverReturn(
                    flag=flag,
                    values=SolverVariables(t=None, y=None, ydot=None),
                    errors=SolverVariables(t=time, y=np_y0, ydot=np_yp0),
                    roots=SolverVariables(t=None, y=None, ydot=None),
                    tstop=SolverVariables(t=None, y=None, ydot=None),
                    message=STATUS_MESSAGE[flag]
                )
        else:
            if self._old_api:
                return (True, time)
            else:
                y_retn  = np.empty(np.alen(np_y0), float)
                yp_retn = np.empty(np.alen(np_y0), float)
                # if no init cond computed, the start values are values at t=0
                y_retn[:] = np_y0[:]
                yp_retn[:] = np_yp0[:]
                if (self.options['compute_initcond'] and
                    (self.options['compute_initcond'].lower() in ['y0', 'yp0'])):
                    #an init cond was computed, obtain it to return it
                    nv_s2ndarray(self.y, y_retn)
                    nv_s2ndarray(self.yp, yp_retn)
                soln = SolverReturn(
                    flag=flag,
                    values=SolverVariables(t=time, y=y_retn, ydot=yp_retn),
                    errors=SolverVariables(t=None, y=None, ydot=None),
                    roots=SolverVariables(t=None, y=None, ydot=None),
                    tstop=SolverVariables(t=None, y=None, ydot=None),
                    message=STATUS_MESSAGE[flag]
                )
        if self._validate_flags:
            return self.validate_flags(soln)
        return soln

    cpdef _init_step(self, DTYPE_t t0,
                     np.ndarray[DTYPE_t, ndim=1] y0,
                     np.ndarray[DTYPE_t, ndim=1] yp0,
                     np.ndarray y_ic0_retn = None,
                     np.ndarray yp_ic0_retn = None):
        """
            Applies the set by 'set_options' method to the IDA solver.

            Performs computation of initial conditions (if compute_initcond
            flag set). Used only in conjuction with the 'step' method to provide
            initialization and assure that the initial condition is calculated.

            Input:
                y0 - initial condition for y
                yp0 - initial condition for derivation of y
                y_ic0_retn  - (optional) returns the calculated consistent
                              initial condition for y
                yp_ic0_retn - (optional) returns the calculated consistent
                              initial condition for y derivated

            Note: After setting (or changing) options with 'set_options' method
                  you need to call 'init_step' prior to the 'step' method to
                  assure the values are correctly initialized.

            Returns the (flag, time) of the intial step.
        """
        #TODO: jacobi function ?isset

        cdef dict opts = self.options

        self.parallel_implementation = (opts['implementation'].lower() == 'parallel')
        if self.parallel_implementation:
            raise ValueError('Error: Parallel implementation not implemented !')
        cdef long int N
        N = <long int> np.alen(y0)

        if opts['rfn'] is None:
            raise ValueError('The residual function rfn not assigned '
                              'during ''set_options'' call !')

        if not np.alen(y0) == np.alen(yp0):
            raise ValueError('Arrays inconsistency: y0 and ydot0 have to be of '
                             'the same length !')
        if (((np.alen(y0) == 0) and (not opts['compute_initcond'] == 'y0'))
                or ((np.alen(yp0) == 0) and (not opts['compute_initcond'] == 'yp0'))):
            raise ValueError('Value of y0 not set or the value of ydot0 not '
                             'flagged to be computed (see ''init_cond'' for '
                             'documentation.')

        #TODO: when implementing parallel, does N_VDestroy be called separately
        #      for parallel version or it's a generic one?
        if not self.y0 is NULL:
            N_VDestroy(self.y0)
            N_VDestroy(self.yp0)
            N_VDestroy(self.residual)
            N_VDestroy(self.y)
            N_VDestroy(self.yp)

        if self.parallel_implementation:
            raise NotImplemented
        else:
            self.y0  = N_VMake_Serial(N, <realtype *>y0.data)
            self.yp0 = N_VMake_Serial(N, <realtype *>yp0.data)
            self.residual = N_VNew_Serial(N)
            self.y   = N_VClone(self.y0)  #clone does not copy data!
            self.yp  = N_VClone(self.yp0) #clone does not copy data!


        cdef int flag
        cdef void* ida_mem = self._ida_mem

        if (ida_mem is NULL) or (self.N != N):
            if (not ida_mem is NULL):
                IDAFree(&ida_mem)
            ida_mem = IDACreate()
            if ida_mem is NULL:
                raise MemoryError('IDACreate:MemoryError: Could not allocate '
                                  'ida_mem object')

            self._ida_mem = ida_mem
            flag = IDAInit(ida_mem, _res, <realtype> t0, self.y0, self.yp0)
        elif self.N == N:
            flag = IDAReInit(ida_mem, <realtype> t0, self.y0, self.yp0)
        else:
            raise ValueError('IDAInit:Error: You should not be here...')
        if flag == IDA_ILL_INPUT:
                raise ValueError('IDA[Re]Init: Ill input')
        elif flag == IDA_MEM_FAIL:
            raise MemoryError('IDA[Re]Init: Memory allocation error')
        elif flag == IDA_MEM_NULL:
            raise MemoryError('IDACreate: Memory allocation error')
        elif flag == IDA_NO_MALLOC:
            raise MemoryError('IDAReInit: No memory allocated in IDAInit.')

        self.N = N

        # auxiliary variables
        self.aux_data = IDA_data(N)

        # Set err_handler
        err_handler = opts.get('err_handler', None)
        if err_handler is not None:
            if not isinstance(err_handler, IDA_ErrHandler):
                tmpfun = IDA_WrapErrHandler()
                tmpfun.set_err_handler(err_handler)
                err_handler = tmpfun

            self.aux_data.err_handler = err_handler

            flag = IDASetErrHandlerFn(
                ida_mem, _ida_err_handler_fn, <void*> self.aux_data)

            if flag == IDA_SUCCESS:
                pass
            elif flag == IDA_MEM_FAIL:
                raise MemoryError(
                    'IDASetErrHandlerFn: Memory allocation error')
            else:
                raise RuntimeError('IDASetErrHandlerFn: Unknown flag raised')
        self.aux_data.err_user_data = opts['err_user_data'] or opts['user_data']

        self.aux_data.parallel_implementation = self.parallel_implementation

        rfn = opts['rfn']
        if not isinstance(rfn , IDA_RhsFunction):
            tmpfun = IDA_WrapRhsFunction()
            tmpfun.set_resfn(rfn)
            rfn = tmpfun
            opts['rfn'] = tmpfun
        self.aux_data.res = rfn

        jac = opts['jacfn']
        if jac is not None and not isinstance(jac , IDA_JacRhsFunction):
            tmpfun = IDA_WrapJacRhsFunction()
            tmpfun.set_jacfn(jac)
            jac = tmpfun
            opts['jacfn'] = tmpfun
        self.aux_data.jac = jac
        self.aux_data.user_data = opts['user_data']

        self._set_runtime_changeable_options(opts, supress_supported_check=True)

        if flag == IDA_ILL_INPUT:
            raise ValueError('IDAStolerances: negative atol or rtol value.')

        # As user data we pass the IDA_data object
        IDASetUserData(ida_mem, <void*> self.aux_data)

        if (opts['order'] > 0):
            IDASetMaxOrd(ida_mem, <int> opts['order'])
        IDASetMaxNumSteps(ida_mem, <int> opts['max_steps'])
        if opts['first_step_size'] > 0.:
            IDASetInitStep(ida_mem, <realtype> opts['first_step_size'])
        if opts['max_step_size'] > 0.:
            flag = IDASetMaxStep(ida_mem, <realtype> opts['max_step_size'])
            if flag == IDA_ILL_INPUT:
                raise ValueError('IDASetMaxStep: max_step_size is negative or '
                                 'smaller than allowed.')
        if opts['max_nonlin_iters'] > 0:
            IDASetMaxNonlinIters(ida_mem, <int> opts['max_nonlin_iters'])
        if opts['max_conv_fails'] > 0:
            IDASetMaxConvFails(ida_mem, <int> opts['max_conv_fails'])
        if opts['nonlin_conv_coef'] > 0:
            IDASetNonlinConvCoef(ida_mem, <int> opts['nonlin_conv_coef'])

        # Linsolver
        linsolver = opts['linsolver'].lower()

        if linsolver == 'dense':
            if self.parallel_implementation:
                raise ValueError('Linear solver for dense matrices can be used '
                                  'only for serial implementation. For parallel'
                                  ' implementation use ''lapackdense'' instead.')
            else:
                 flag = IDADense(ida_mem, N)
                 if flag == IDADLS_ILL_INPUT:
                     raise ValueError('IDADense solver is not compatible with'
                                      ' the current nvector implementation.')
                 elif flag == IDADLS_MEM_FAIL:
                     raise MemoryError('IDADense memory allocation error.')
        elif linsolver == 'lapackdense':
            flag = IDALapackDense(ida_mem, N)
            if flag == IDADLS_ILL_INPUT:
                raise ValueError('IDALapackDense solver is not compatible with'
                                 ' the current nvector implementation.')
            elif flag == IDADLS_MEM_FAIL:
                raise MemoryError('IDALapackDense memory allocation error.')
        elif linsolver == 'band':
            if self.parallel_implementation:
                raise ValueError('Linear solver for band matrices can be used'
                                 'only for serial implementation. For parallel'
                                 ' implementation use ''lapackband'' instead.')
            else:
                flag = IDABand(ida_mem, N, <int> opts['uband'],
                                           <int> opts['lband'])
                if flag == IDADLS_ILL_INPUT:
                    raise ValueError('IDABand solver is not compatible'
                                     ' with the current nvector implementation'
                                     ' or bandwith outside range.')
                elif flag == IDADLS_MEM_FAIL:
                    raise MemoryError('IDABand memory allocation error.')
        elif linsolver == 'lapackband':
            flag = IDALapackBand(ida_mem, N, <int> opts['uband'],
                                             <int> opts['lband'])
            if flag == IDADLS_ILL_INPUT:
                raise ValueError('IDALapackBand solver is not compatible'
                                 ' with the current nvector implementation'
                                 ' or bandwith outside range.')
            elif flag == IDADLS_MEM_FAIL:
                raise MemoryError('IDALapackBand memory allocation error.')
        elif ((linsolver == 'spgmr') or (linsolver == 'spbcg')
                  or (linsolver == 'sptfqmr')):
            maxl = <int> opts['maxl']

            if linsolver == 'spgmr':
                flag = IDASpgmr(ida_mem, maxl)
            elif linsolver == 'spbcg':
                flag = IDASpbcg(ida_mem, maxl)
            else:
                flag = IDASptfqmr(ida_mem, maxl)

            if flag == IDASPILS_MEM_FAIL:
                raise MemoryError('LinSolver:IDASpils memory allocation error.')

        if (linsolver in ['dense', 'lapackdense']) and self.aux_data.jac:
            IDADlsSetDenseJacFn(ida_mem, _jacdense)

        # Constraints
        constraints_idx  = opts['constraints_idx']
        constraints_type = opts['constraints_type']
        cdef unsigned int idx
        cdef np.ndarray[DTYPE_t, ndim=1] constraints_vars

        if not constraints_type is None:
            if not constraints_idx is None:
                constraints_vars = np.zeros(N, float)
                for idx in range(constraints_idx):
                    constraints_vars[constraints_idx[idx]] = constraints_type[idx]
            else:
                assert np.alen(constraints_type) == N,\
                  'Without ''constraints_idx'' specified the'\
                  '  ''constraints_type'' the constraints_idx has to be of the'\
                  ' same length as y (i.e. contains constraints for EVERY '\
                  'component of y).'
                constraints_vars = np.asarray(constraints_type, float)

            if not self.constraints is NULL:
                N_VDestroy(self.constraints)
                self.constraints = NULL

            if self.parallel_implementation:
                raise NotImplemented
            else:
                self.constraints =\
                   N_VMake_Serial(N, <realtype*> constraints_vars.data)
            flag = IDASetConstraints(ida_mem, self.constraints)
            if flag == IDA_ILL_INPUT:
                raise ValueError('IDAConstraints: ''constraints_type'' contains'
                                 'illegal value.')

        # DAE variables and Initial condition
        cdef np.ndarray[DTYPE_t, ndim=1] dae_vars

        alg_vars_idx     = opts['algebraic_vars_idx']

        if opts['compute_initcond']:
            compute_initcond = opts['compute_initcond'].lower()
        else:
            compute_initcond = ''

        if ((compute_initcond == 'yp0') or
            (opts['exclude_algvar_from_error'] and not alg_vars_idx is None)):

            # DAE variables
            dae_vars = np.ones(N, float)

            if not alg_vars_idx is None:
                for algvar_idx in opts['algebraic_vars_idx']:
                    dae_vars[algvar_idx] = 0.0

            if not self.dae_vars_id is NULL:
                N_VDestroy(self.dae_vars_id)
                self.dae_vars_id = NULL

            if self.parallel_implementation:
                raise NotImplemented
            else:
                self.dae_vars_id = N_VMake_Serial(N, <realtype*> dae_vars.data)

            IDASetId(ida_mem, self.dae_vars_id)

            if opts['exclude_algvar_from_error'] and not alg_vars_idx is None:
                IDASetSuppressAlg(ida_mem, True)

        # Initial condition
        cdef bint compute_initcond_p
        cdef realtype ic_t0 = <realtype>opts['compute_initcond_t0']

        if compute_initcond in [None, 'y0', 'yp0', '']:
            compute_initcond_p = (compute_initcond == 'y0' or
                                  compute_initcond == 'yp0')
        else:
            raise ValueError('InitCond: Unknown ''compute_initcond'' value: %s'
                             % compute_initcond)

        #now we initialize storage which is persistent over steps
        self.t_roots = []
        self.y_roots = []
        self.yp_roots = []
        self.t_tstop = []
        self.y_tstop = []
        self.yp_tstop = []

        yinit = self.y0
        ypinit = self.yp0
        if compute_initcond_p:
            flag = IDA_ILL_INPUT

            if compute_initcond == 'yp0':
                flag = IDACalcIC(ida_mem, IDA_YA_YDP_INIT, t0 + ic_t0)
            elif compute_initcond == 'y0':
                flag = IDACalcIC(ida_mem, IDA_Y_INIT, t0 + ic_t0)

            if not (flag == IDA_SUCCESS):
                return (flag, t0)
            else:
                flag = IDAGetConsistentIC(self._ida_mem, self.y, self.yp)

                if flag == IDA_ILL_INPUT:
                    raise NameError('Method ''get_consistent_ic'' has to be called'
                                    ' prior the first call to the ''step'' method.')

                yinit = self.y
                ypinit = self.yp

        if not ((y_ic0_retn is None) and (yp_ic0_retn is None)):
            assert np.alen(y_ic0_retn) == np.alen(yp_ic0_retn) == N,\
              'y_ic0 and/or yp_ic0 have to be of the same size as y0.'

            if not y_ic0_retn is None: nv_s2ndarray(yinit, y_ic0_retn)
            if not yp_ic0_retn is None: nv_s2ndarray(ypinit, yp_ic0_retn)

        #TODO: implement the rest of IDA*Set* functions for linear solvers

        #TODO: Reinitialization of the rooting function

        self.initialized = True

        return (flag, t0)

    def reinit_IC(self, double t0, object y0, object yp0):
        """
        Re-initialize (only) the initial condition IC without re-setting also
        all the remaining solver options. See also 'init_step()' funtion.
        This does not allow to recompute a valid IC condition, use init_step for that.
        Typically, an exception is raised with wrong input.

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
        cdef np.ndarray[DTYPE_t, ndim=1] np_y0
        np_y0 = np.asarray(y0)
        cdef np.ndarray[DTYPE_t, ndim=1] np_yp0
        np_yp0 = np.asarray(yp0)

        flag, time = self._reinit_IC(t0, np_y0, np_yp0)

        if self._old_api:
            return (flag, time)
        else:
            y_retn  = np.empty(np.alen(np_y0), float)
            yp_retn = np.empty(np.alen(np_y0), float)
            # no init cond computed, the start values are values at t=t0
            y_retn[:] = np_y0[:]
            yp_retn[:] = np_yp0[:]
            soln = SolverReturn(
                flag=flag,
                values=SolverVariables(t=time, y=y_retn, ydot=yp_retn),
                errors=SolverVariables(t=None, y=None, ydot=None),
                roots=SolverVariables(t=None, y=None, ydot=None),
                tstop=SolverVariables(t=None, y=None, ydot=None),
                message=STATUS_MESSAGE[StatusEnumIDA.SUCCESS]
            )
        if self._validate_flags:
            return self.validate_flags(soln)
        return soln

    cpdef _reinit_IC(self, double t0, np.ndarray[DTYPE_t, ndim=1] y0,
                     np.ndarray[DTYPE_t, ndim=1] yp0):
        # If not yet initialized, run full initialization
        if self.y0 is NULL:
            self._init_step(t0, y0, yp0)
            return

        cdef long int N
        N = <long int> np.alen(y0)
        Np = <long int> np.alen(yp0)
        if N == self.N and Np == N:
            self.y0  = N_VMake_Serial(N, <realtype *>y0.data)
            self.yp0  = N_VMake_Serial(N, <realtype *>yp0.data)
        else:
            raise ValueError("Cannot re-init IC with array of unequal lenght.")

        flag = IDAReInit(self._ida_mem, <realtype> t0, self.y0, self.yp0)

        if flag == IDA_ILL_INPUT:
                raise ValueError('IDA[Re]Init: Ill input')
        elif flag == IDA_MEM_FAIL:
            raise MemoryError('IDA[Re]Init: Memory allocation error')
        elif flag == IDA_MEM_NULL:
            raise MemoryError('IDACreate: Memory allocation error')
        elif flag == IDA_NO_MALLOC:
            raise MemoryError('IDAReInit: No memory allocated in IDAInit.')

        return (True, t0)

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

        cdef np.ndarray[DTYPE_t, ndim=1] np_tspan, np_y0, np_yp0

        np_tspan = np.asarray(tspan, dtype=float)
        np_y0    = np.asarray(y0, dtype=float)
        np_yp0   = np.asarray(yp0, dtype=float)

        soln = self._solve(np_tspan, np_y0, np_yp0)
        if self._old_api:
            warn("Old api is deprecated, move to new api", DeprecationWarning)
            return soln

        if self._validate_flags:
            return self.validate_flags(soln)
        return soln

    cpdef _solve(self, np.ndarray[DTYPE_t, ndim=1] tspan,
                       np.ndarray[DTYPE_t, ndim=1] y0,
                       np.ndarray[DTYPE_t, ndim=1] yp0):

        cdef np.ndarray[DTYPE_t, ndim=1] t_retn
        cdef np.ndarray[DTYPE_t, ndim=2] y_retn, yp_retn
        t_retn  = np.empty([np.alen(tspan), ], float)
        y_retn  = np.empty([np.alen(tspan), np.alen(y0)], float)
        yp_retn = np.empty([np.alen(tspan), np.alen(y0)], float)

        cdef int flag

        #check to avoid typical error
        cdef dict opts = self.options
        cdef realtype ic_t0 = <realtype>opts['compute_initcond_t0']
        if ((np.alen(tspan)>1 and ic_t0 > 0. and tspan[1] > tspan[0]) or
            (np.alen(tspan)>1 and ic_t0 < 0. and tspan[1] < tspan[0])):
            pass
        else:
            raise ValueError('InitCond: ''compute_initcond_t0'' value: %f '
                    ' is in different direction from start time %f than the '
                    'first required output %f.'
                             % (ic_t0, tspan[0], tspan[1]))

        ret_ic = self.init_step(tspan[0], y0, yp0,
                                           y_retn[0, :], yp_retn[0, :])

        PyErr_CheckSignals()
        if self._old_api:
            flag = ret_ic[0]
        else:
            flag = (ret_ic.flag == IDA_SUCCESS)

        if not flag:
            if self._old_api:
                # print done in init_step method!
#                print('IDAInitCond: Error occured during computation'
#                      ' of initial condition, flag', flag)
                return (False, ret_ic[1], y0, None, None, None, None)
            else:
                return ret_ic
        else:
            if self._old_api:
                t_retn[0] = ret_ic[1]
            else:
                t_retn[0] = ret_ic.values.t
        #TODO: Parallel version
        cdef np.ndarray[DTYPE_t, ndim=1] y_last, yp_last
        cdef unsigned int idx = 1 # idx == 0 is IC
        cdef unsigned int last_idx = np.alen(tspan)
        cdef DTYPE_t t
        cdef void *ida_mem = self._ida_mem
        cdef realtype t_out
        cdef N_Vector y  = self.y
        cdef N_Vector yp = self.yp
        cdef IDA_ContinuationFunction onroot = self.options['onroot']
        cdef IDA_ContinuationFunction ontstop = self.options['ontstop']

        y_last   = np.empty(np.shape(y0), float)
        yp_last  = np.empty(np.shape(y0), float)


        t = tspan[idx]

        while True:
            flag = IDASolve(self._ida_mem, <realtype> t, &t_out, y, yp,
                            IDA_NORMAL)

            nv_s2ndarray(y,  y_last)
            nv_s2ndarray(yp,  yp_last)

            if flag == IDA_SUCCESS or flag == IDA_WARNING:
                t_retn[idx]     = t_out
                y_retn[idx, :]  = y_last
                yp_retn[idx, :] = yp_last
                idx = idx + 1
                PyErr_CheckSignals()

                # Iterate until we reach the end of tspan
                if idx < last_idx:
                    t = tspan[idx]
                    continue
                else:
                    break

            elif flag == IDA_ROOT_RETURN:
                self.t_roots.append(np.copy(t_out))
                self.y_roots.append(np.copy(y_last))
                self.yp_roots.append(np.copy(yp_last))
                root_flag = onroot.evaluate(t_out, y_last, yp_last, self)
                if (t_out == t):
                    #a root at our wanted output, and we continue comp
                    t_retn[idx]    = t_out
                    y_retn[idx, :] = y_last
                    yp_retn[idx, :] = yp_last
                    idx = idx + 1
                    if idx < last_idx:
                        t = tspan[idx]
                    else:
                        break
                if root_flag == 0:
                    PyErr_CheckSignals()
                    continue
                break
            elif flag == IDA_TSTOP_RETURN:
                self.t_tstop.append(np.copy(t_out))
                self.y_tstop.append(np.copy(y_last))
                self.yp_tstop.append(np.copy(yp_last))
                tstop_flag = ontstop.evaluate(t_out, y_last, yp_last, self)
                if (t_out == t):
                    #a tstop at our wanted output, and we continue comp
                    t_retn[idx]    = t_out
                    y_retn[idx, :] = y_last
                    yp_retn[idx, :] = yp_last
                    idx = idx + 1
                    if idx < last_idx:
                        t = tspan[idx]
                    else:
                        break
                if tstop_flag == 0:
                    PyErr_CheckSignals()
                    continue
                break

            break


        PyErr_CheckSignals()

        # return values computed so far
        t_retn  = t_retn[0:idx]
        y_retn  = y_retn[0:idx, :]
        yp_retn  = yp_retn[0:idx, :]
        if flag < 0:
            y_err = np.copy(y_last)
            yp_err = np.copy(yp_last)
            t_err = t_out
            soln = flag, t_retn, y_retn, yp_retn, t_out, y_last, yp_last
        else:
            y_err = None
            t_err = None
            yp_err   = None
            soln = flag, t_retn, y_retn, yp_retn, None, None, None

        if self._old_api:
            return soln

        t_roots = np.array(self.t_roots) if self.t_roots else None
        y_roots = np.array(self.y_roots) if self.y_roots else None
        yp_roots = np.array(self.yp_roots) if self.yp_roots else None
        t_tstop = np.array(self.t_tstop) if self.t_tstop else None
        y_tstop = np.array(self.y_tstop) if self.y_tstop else None
        yp_tstop = np.array(self.yp_tstop) if self.yp_tstop else None

        return SolverReturn(
            flag=flag,
            values=SolverVariables(t=t_retn, y=y_retn, ydot=yp_retn),
            errors=SolverVariables(t=t_err, y=y_err, ydot=yp_err),
            roots=SolverVariables(t=t_roots, y=y_roots, ydot=yp_roots),
            tstop=SolverVariables(t=t_tstop, y=y_tstop, ydot=yp_tstop),
            message=STATUS_MESSAGE[flag]
        )

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
        if not self.initialized:
            raise ValueError('Method ''init_step'' has to be called prior to'
                             'the first call of ''step'' method.')
        cdef N_Vector y  = self.y
        cdef N_Vector yp = self.yp
        cdef realtype t_out
        cdef int flagIDA
        if self._old_api:
            if t>0.0:
                flagIDA = IDASolve(self._ida_mem, <realtype> t, &t_out, y, yp,
                                   IDA_NORMAL)
            else:
                flagIDA = IDASolve(self._ida_mem, <realtype> -t, &t_out, y, yp,
                                   IDA_ONE_STEP)
        else:
            if self._step_compute:
                flagIDA = IDASolve(self._ida_mem, <realtype> t, &t_out, y, yp,
                                   IDA_ONE_STEP)
            else:
                flagIDA = IDASolve(self._ida_mem, <realtype> t, &t_out, y, yp,
                                   IDA_NORMAL)

        cdef np.ndarray[DTYPE_t, ndim=1] y_out
        cdef np.ndarray[DTYPE_t, ndim=1] yp_out
        if not y_retn is None:
            nv_s2ndarray(y, y_retn)
            y_out = y_retn
        else:
            y_out  = np.empty(self.N, float)
            nv_s2ndarray(y, y_out)

        if not yp_retn is None:
            nv_s2ndarray(yp, yp_retn)
            yp_out = yp_retn
        else:
            yp_out  = np.empty(self.N, float)
            nv_s2ndarray(yp, yp_out)

        flag = StatusEnumIDA(flagIDA)

        t_err = None
        y_err = None
        yp_err = None
        sol_t_out = t_out
        if flag == IDA_SUCCESS or flag == IDA_WARNING:
            pass
        elif flag == IDA_ROOT_RETURN:
            self.t_roots.append(np.copy(t_out))
            self.y_roots.append(np.copy(y_out))
            self.yp_roots.append(np.copy(yp_out))
        elif flag == IDA_TSTOP_RETURN:
            self.t_tstop.append(np.copy(t_out))
            self.y_tstop.append(np.copy(y_out))
            self.yp_tstop.append(np.copy(yp_out))
        elif flag < 0:
            t_err = np.copy(t_out)
            y_err = np.copy(y_out)
            yp_err = np.copy(yp_out)
            sol_t_out = None
            y_out = None
            yp_out = None

        PyErr_CheckSignals()

        t_roots = np.array(self.t_roots) if self.t_roots else None
        y_roots = np.array(self.y_roots) if self.y_roots else None
        yp_roots = np.array(self.yp_roots) if self.yp_roots else None
        t_tstop = np.array(self.t_tstop) if self.t_tstop else None
        y_tstop = np.array(self.y_tstop) if self.y_tstop else None
        yp_tstop = np.array(self.yp_tstop) if self.yp_tstop else None

        if self._old_api:
            return flagIDA, t_out

        return SolverReturn(
            flag=flag,
            values=SolverVariables(t=sol_t_out, y=y_out, ydot=yp_out),
            errors=SolverVariables(t=t_err, y=y_err, ydot=yp_err),
            roots=SolverVariables(t=t_roots, y=y_roots, ydot=yp_roots),
            tstop=SolverVariables(t=t_tstop, y=y_tstop, ydot=yp_tstop),
            message=STATUS_MESSAGE[flag]
        )

    def validate_flags(self, soln):
        """
        Validates the flag returned by `IDA.solve`.

        Validation happens using the following scheme:
         * failures (`flag` < 0) raise `IDASolveFailed` or a subclass of it;
         * finding a root (and stopping) raises `IDASolveFoundRoot`;
         * reaching `tstop` (and stopping) raises `IDASolveReachedTSTOP`;
         * otherwise, return an instance of `SolverReturn`.

        In the case where ontstop or onroot are used, `IDASolveFoundRoot` or
        `IDASolveReachedTSTOP` will be raised only if the solver is told to
        stop at that point.
        """
        if soln.flag == StatusEnumIDA.SUCCESS:
            return soln
        if soln.flag < 0:
            raise IDASolveFailed(soln)
        elif soln.flag == StatusEnumIDA.TSTOP_RETURN:
            raise IDASolveReachedTSTOP(soln)
        elif soln.flag == StatusEnumIDA.ROOT_RETURN:
            raise IDASolveFoundRoot(soln)
        warn(WARNING_STR.format(soln.flag, *soln.err_values))
        return soln

    def __dealloc__(self):
        if not self._ida_mem is NULL: IDAFree(&self._ida_mem)
        #TODO: when implementing parallel, does N_VDestroy be called separately
        #      for parallel version or it's a generic one?
        if not self.atol is NULL: N_VDestroy(self.atol)
        if not self.y0 is NULL: N_VDestroy(self.y0)
        if not self.yp0 is NULL: N_VDestroy(self.yp0)
        if not self.y is NULL: N_VDestroy(self.y)
        if not self.yp is NULL: N_VDestroy(self.yp)
        if not self.residual is NULL: N_VDestroy(self.residual)
        if not self.dae_vars_id is NULL: N_VDestroy(self.dae_vars_id)
        if not self.constraints is NULL: N_VDestroy(self.constraints)
