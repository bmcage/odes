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

from .c_cvode cimport *
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
    StatusEnum.UNRECOGNIZED_ERR: "Unrecognized Error",
}

WARNING_STR = "Solver succeeded with flag {} and finished at {} with values {}"

# Right-hand side function
cdef class CV_RhsFunction:
    """
    Prototype for rhs function.

    Note that evaluate must return a integer, 0 for success, positive for
    recoverable failure, negative for unrecoverable failure (as per CVODE
    documentation).
    """
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       object userdata = None) except? -1:
        return 0

cdef class CV_WrapRhsFunction(CV_RhsFunction):

    cpdef set_rhsfn(self, object rhsfn):
        """
        set some rhs equations as a RhsFunction executable class
        """
        self.with_userdata = 0
        nrarg = _get_num_args(rhsfn)
        if nrarg > 4:
            #hopefully a class method, self gives 5 arg!
            self.with_userdata = 1
        elif nrarg == 4 and inspect.isfunction(rhsfn):
            self.with_userdata = 1
        self._rhsfn = rhsfn

    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       object userdata = None) except? -1:
        if self.with_userdata == 1:
            user_flag = self._rhsfn(t, y, ydot, userdata)
        else:
            user_flag = self._rhsfn(t, y, ydot)
        if user_flag is None:
            user_flag = 0
        return user_flag

cdef int _rhsfn(realtype tt, N_Vector yy, N_Vector yp,
              void *auxiliary_data) except? -1:
    """ function with the signature of CVRhsFn, that calls python Rhs """
    cdef np.ndarray[DTYPE_t, ndim=1] yy_tmp, yp_tmp

    aux_data = <CV_data> auxiliary_data
    cdef bint parallel_implementation = aux_data.parallel_implementation

    if parallel_implementation:
        raise NotImplemented
    else:
        yy_tmp = aux_data.yy_tmp
        yp_tmp = aux_data.yp_tmp

        nv_s2ndarray(yy, yy_tmp)
        #nv_s2ndarray(yp, yp_tmp)

    user_flag = aux_data.rfn.evaluate(tt, yy_tmp, yp_tmp, aux_data.user_data)

    if parallel_implementation:
        raise NotImplemented
    else:
        ndarray2nv_s(yp, yp_tmp)

    return user_flag

# Root function
cdef class CV_RootFunction:
    """
    Prototype for root function.

    Note that evaluate must return a integer, 0 for success, non-zero if error
    (as per CVODE documentation).
    """
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] g,
                       object userdata = None) except? -1:
        return 0

cdef class CV_WrapRootFunction(CV_RootFunction):
    cpdef set_rootfn(self, object rootfn):
        """
        set root-ing condition(equations) as a RootFunction executable class
        """
        self.with_userdata = 0
        nrarg = _get_num_args(rootfn)
        if nrarg > 4:
            #hopefully a class method, self gives 4 arg!
            self.with_userdata = 1
        elif nrarg == 4 and inspect.isfunction(rootfn):
            self.with_userdata = 1
        self._rootfn = rootfn

    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] g,
                       object userdata = None) except? -1:
        if self.with_userdata == 1:
            user_flag = self._rootfn(t, y, g, userdata)
        else:
            user_flag = self._rootfn(t, y, g)
        if user_flag is None:
            user_flag = 0
        return user_flag

cdef int _rootfn(realtype t, N_Vector y, realtype *gout, void *auxiliary_data) except? -1:
    """ function with the signature of CVRootFn """

    aux_data = <CV_data> auxiliary_data
    cdef bint parallel_implementation = aux_data.parallel_implementation

    if parallel_implementation:
        raise NotImplemented
    else:
        yy_tmp = aux_data.yy_tmp
        g_tmp  = aux_data.g_tmp

        nv_s2ndarray(y, yy_tmp)

    user_flag = aux_data.rootfn.evaluate(t, yy_tmp, g_tmp, aux_data.user_data)

    cdef int i
    if parallel_implementation:
        raise NotImplemented
    else:
        for i in np.arange(np.alen(g_tmp)):
            gout[i] = <realtype> g_tmp[i]

    return user_flag

# Jacobian function
cdef class CV_JacRhsFunction:
    """
    Prototype for jacobian function.

    Note that evaluate must return a integer, 0 for success, positive for
    recoverable failure, negative for unrecoverable failure (as per CVODE
    documentation).
    """
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] fy,
                       np.ndarray[DTYPE_t, ndim=2] J) except? -1:
        """
        Returns the Jacobi matrix of the right hand side function, as
            d(rhs)/d y
        (for dense the full matrix, for band only bands). Result has to be
        stored in the variable J, which is preallocated to the corresponding
        size.

        This is a generic class, you should subclass is for the problem specific
        purposes.
        """
        return 0

cdef class CV_WrapJacRhsFunction(CV_JacRhsFunction):
    cpdef set_jacfn(self, object jacfn):
        """
        Set some jacobian equations as a JacRhsFunction executable class.
        """
        self._jacfn = jacfn

    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] fy,
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
        user_flag = self._jacfn(t, y, fy, J)

        if user_flag is None:
            user_flag = 0
        return user_flag

cdef int _jacdense(realtype tt,
            N_Vector yy, N_Vector ff, SUNMatrix Jac,
            void *auxiliary_data, N_Vector tmp1, N_Vector tmp2, 
            N_Vector tmp3) except? -1:
    """function with the signature of CVDlsJacFn that calls python Jac
       Note: signature of Jac is SUNMatrix
    """
    cdef np.ndarray[DTYPE_t, ndim=1] yy_tmp, ff_tmp
    cdef np.ndarray[DTYPE_t, ndim=2] jac_tmp

    aux_data = <CV_data> auxiliary_data
    cdef bint parallel_implementation = aux_data.parallel_implementation
    if parallel_implementation:
        raise NotImplemented
    else:
        yy_tmp = aux_data.yy_tmp
        jac_tmp = aux_data.jac_tmp

        nv_s2ndarray(yy, yy_tmp)
        ff_tmp = aux_data.z_tmp
        nv_s2ndarray(ff, ff_tmp)

    user_flag = aux_data.jac.evaluate(tt, yy_tmp, ff_tmp, jac_tmp)

    if parallel_implementation:
        raise NotImplemented
    else:
        ndarray2SUNMatrix(Jac, jac_tmp)

    return user_flag

# Precondioner setup funtion
cdef class CV_PrecSetupFunction:
    """
    Prototype for preconditioning setup function.

    Note that evaluate must return a integer, 0 for success, positive for
    recoverable failure, negative for unrecoverable failure (as per CVODE
    documentation).
    """
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       bint jok,
                       object jcurPtr,
                       DTYPE_t gamma,
                       object userdata = None) except? -1:
        """
        This function preprocesses and/or evaluates Jacobian-related data
        needed by the preconditioner. Use the userdata object to expose
        the preprocessed data to the solve function.

        This is a generic class, you should subclass it for the problem specific
        purposes.
        """
        return 0

cdef class CV_WrapPrecSetupFunction(CV_PrecSetupFunction):
    cpdef set_prec_setupfn(self, object prec_setupfn):
        """
        set a precondititioning setup method as a CV_PrecSetupFunction
        executable class
        """
        self.with_userdata = 0
        nrarg = _get_num_args(prec_setupfn)
        if nrarg > 6:
            #hopefully a class method, self gives 7 arg!
            self.with_userdata = 1
        elif nrarg == 6 and inspect.isfunction(prec_setupfn):
            self.with_userdata = 1
        self._prec_setupfn = prec_setupfn

    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       bint jok,
                       object jcurPtr,
                       DTYPE_t gamma,
                       object userdata = None) except? -1:
        if self.with_userdata == 1:
            user_flag = self._prec_setupfn(t, y, jok, jcurPtr, gamma, userdata)
        else:
            user_flag = self._prec_setupfn(t, y, jok, jcurPtr, gamma)
        if user_flag is None:
            user_flag = 0
        return user_flag

class MutableBool(object):
    def __init__(self, value):
        self.value = value

cdef int _prec_setupfn(realtype tt, N_Vector yy, N_Vector ff, booleantype jok, 
                       booleantype *jcurPtr, realtype gamma, 
                       void *auxiliary_data) except -1:
    """ function with the signature of CVLsPrecSetupFn, that calls python function """
    cdef np.ndarray[DTYPE_t, ndim=1] yy_tmp

    aux_data = <CV_data> auxiliary_data
    cdef bint parallel_implementation = aux_data.parallel_implementation

    if parallel_implementation:
        raise NotImplemented
    else:
        yy_tmp = aux_data.yy_tmp
        nv_s2ndarray(yy, yy_tmp)

    jcurPtr_tmp = MutableBool(jcurPtr[0])
    user_flag = aux_data.prec_setupfn.evaluate(tt, yy_tmp, jok, jcurPtr_tmp, 
                                               gamma, aux_data.user_data)
    jcurPtr[0] = jcurPtr_tmp.value
    return user_flag

# Precondioner solve funtion
cdef class CV_PrecSolveFunction:
    """
    Prototype for precondititioning solution function.

    Note that evaluate must return a integer, 0 for success, positive for
    recoverable failure, negative for unrecoverable failure (as per CVODE
    documentation).
    """
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] r,
                       np.ndarray[DTYPE_t, ndim=1] z,
                       DTYPE_t gamma,
                       DTYPE_t delta,
                       int lr,
                       object userdata = None) except? -1:
        """
        This function solves the preconditioned system P*z = r, where P may be
        either a left or right preconditioner matrix. Here P should approximate
        (at least crudely) the Newton matrix M = I − gamma*J, where J is the
        Jacobian of the system. If preconditioning is done on both sides,
        the product of the two preconditioner matrices should approximate M.

        This is a generic class, you should subclass it for the problem specific
        purposes.
        """
        return 0

cdef class CV_WrapPrecSolveFunction(CV_PrecSolveFunction):
    cpdef set_prec_solvefn(self, object prec_solvefn):
        """
        set a precondititioning solve method as a CV_PrecSolveFunction
        executable class
        """
        self.with_userdata = 0
        nrarg = _get_num_args(prec_solvefn)
        if nrarg > 8:
            #hopefully a class method, self gives 9 arg!
            self.with_userdata = 1
        elif nrarg == 8 and inspect.isfunction(prec_solvefn):
            self.with_userdata = 1
        self._prec_solvefn = prec_solvefn

    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] r,
                       np.ndarray[DTYPE_t, ndim=1] z,
                       DTYPE_t gamma,
                       DTYPE_t delta,
                       int lr,
                       object userdata = None) except? -1:
        if self.with_userdata == 1:
            user_flag = self._prec_solvefn(t, y, r, z, gamma, delta, lr, userdata)
        else:
            user_flag = self._prec_solvefn(t, y, r, z, gamma, delta, lr)

        if user_flag is None:
            user_flag = 0
        return user_flag

cdef int _prec_solvefn(realtype tt, N_Vector yy, N_Vector ff, N_Vector r, 
                       N_Vector z, realtype gamma, realtype delta, int lr, 
                       void *auxiliary_data) except? -1:
    """ function with the signature of CVLsPrecSolveFn, that calls python function """
    cdef np.ndarray[DTYPE_t, ndim=1] yy_tmp, r_tmp, z_tmp

    aux_data = <CV_data> auxiliary_data
    cdef bint parallel_implementation = aux_data.parallel_implementation

    if parallel_implementation:
        raise NotImplemented
    else:
        yy_tmp = aux_data.yy_tmp

        r_tmp = aux_data.r_tmp
        z_tmp = aux_data.z_tmp

        nv_s2ndarray(yy, yy_tmp)
        nv_s2ndarray(r, r_tmp)

    user_flag = aux_data.prec_solvefn.evaluate(tt, yy_tmp, r_tmp, z_tmp, gamma, delta, lr, aux_data.user_data)

    if parallel_implementation:
        raise NotImplemented
    else:
        ndarray2nv_s(z, z_tmp)

    return user_flag

# JacTimesVec function
cdef class CV_JacTimesVecFunction:
    """
    Prototype for jacobian times vector function.

    Note that evaluate must return a integer, 0 for success, non-zero for error
    (as per CVODE documentation).
    """
    cpdef int evaluate(self,
                       np.ndarray[DTYPE_t, ndim=1] v,
                       np.ndarray[DTYPE_t, ndim=1] Jv,
                       DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       object userdata = None) except? -1:
        """
        This function calculates the product of the Jacobian with a given vector v.
        Use the userdata object to expose Jacobian related data to the solve function.

        This is a generic class, you should subclass it for the problem specific
        purposes.
        """
        return 0

cdef class CV_WrapJacTimesVecFunction(CV_JacTimesVecFunction):
    cpdef set_jac_times_vecfn(self, object jac_times_vecfn):
        """
        Set some CV_JacTimesVecFn executable class.
        """
        """
        set a jacobian-times-vector method as a CV_JacTimesVecFunction
        executable class
        """
        self.with_userdata = 0
        nrarg = _get_num_args(jac_times_vecfn)
        if nrarg > 5:
            #hopefully a class method, self gives 6 arg!
            self.with_userdata = 1
        elif nrarg == 5 and inspect.isfunction(jac_times_vecfn):
            self.with_userdata = 1
        self._jac_times_vecfn = jac_times_vecfn

    cpdef int evaluate(self,
                       np.ndarray[DTYPE_t, ndim=1] v,
                       np.ndarray[DTYPE_t, ndim=1] Jv,
                       DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       object userdata = None) except? -1:
        if self.with_userdata == 1:
            user_flag = self._jac_times_vecfn(v, Jv, t, y, userdata)
        else:
            user_flag = self._jac_times_vecfn(v, Jv, t, y)
        if user_flag is None:
            user_flag = 0
        return user_flag

cdef int _jac_times_vecfn(N_Vector v, N_Vector Jv, realtype t, N_Vector y,
                          N_Vector fy, void *user_data, N_Vector tmp) except? -1:
    """ function with the signature of CVSpilsJacTimesVecFn, that calls python function """
    cdef np.ndarray[DTYPE_t, ndim=1] y_tmp, v_tmp, Jv_tmp

    aux_data = <CV_data> user_data
    cdef bint parallel_implementation = aux_data.parallel_implementation

    if parallel_implementation:
        raise NotImplemented
    else:
        y_tmp = aux_data.yy_tmp

        v_tmp = aux_data.r_tmp
        Jv_tmp = aux_data.z_tmp

        nv_s2ndarray(y, y_tmp)
        nv_s2ndarray(v, v_tmp)

    user_flag = aux_data.jac_times_vecfn.evaluate(v_tmp, Jv_tmp, t, y_tmp, aux_data.user_data)

    if parallel_implementation:
        raise NotImplemented
    else:
        ndarray2nv_s(Jv, Jv_tmp)

    return user_flag

# JacTimesVec function
cdef class CV_JacTimesSetupFunction:
    """
    Prototype for jacobian times setup function.

    Note that evaluate must return a integer, 0 for success, non-zero for error
    (as per CVODE documentation), with >0  a recoverable error (step is retried).
    """
    cpdef int evaluate(self,
                       DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] fy,
                       object userdata = None) except? -1:
        """
        This function calculates the product of the Jacobian with a given vector v.
        Use the userdata object to expose Jacobian related data to the solve function.

        This is a generic class, you should subclass it for the problem specific
        purposes.
        """
        return 0

cdef class CV_WrapJacTimesSetupFunction(CV_JacTimesSetupFunction):
    cpdef set_jac_times_setupfn(self, object jac_times_setupfn):
        """
        Set some CV_JacTimesSetupFn executable class.
        """
        """
        set a jacobian-times-vector method setup as a CV_JacTimesSetupFunction
        executable class
        """
        self.with_userdata = 0
        nrarg = _get_num_args(jac_times_setupfn)
        if nrarg > 4:
            #hopefully a class method, self gives 5 arg!
            self.with_userdata = 1
        elif nrarg == 4 and inspect.isfunction(jac_times_setupfn):
            self.with_userdata = 1
        self._jac_times_setupfn = jac_times_setupfn

    cpdef int evaluate(self,
                       DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] fy,
                       object userdata = None) except? -1:
        if self.with_userdata == 1:
            user_flag = self._jac_times_setupfn(t, y, fy, userdata)
        else:
            user_flag = self._jac_times_setupfn(t, y, fy)
        if user_flag is None:
            user_flag = 0
        return user_flag

cdef int _jac_times_setupfn(realtype t, N_Vector y, N_Vector fy,
                            void *user_data) except? -1:
    """ function with the signature of CVSpilsJacTimesSetupFn, that calls python function """
    cdef np.ndarray[DTYPE_t, ndim=1] y_tmp, fy_tmp

    aux_data = <CV_data> user_data
    cdef bint parallel_implementation = aux_data.parallel_implementation

    if parallel_implementation:
        raise NotImplemented
    else:
        y_tmp = aux_data.yy_tmp

        fy_tmp = aux_data.z_tmp

        nv_s2ndarray(y, y_tmp)
        nv_s2ndarray(fy, fy_tmp)

    user_flag = aux_data.jac_times_setupfn.evaluate(t, y_tmp, fy_tmp, aux_data.user_data)

    #if parallel_implementation:
    #    raise NotImplemented
    #else:
    #    ndarray2nv_s(fy, fy_tmp)

    return user_flag

cdef class CV_ContinuationFunction:
    """
    Simple wrapper for functions called when ROOT or TSTOP are returned.
    """
    def __cinit__(self, fn):
        self._fn = fn
    cpdef int evaluate(self,
                       DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       CVODE solver):
        return self._fn(t, y, solver)

def no_continue_fn(t, y, solver):
    return 1

cdef class CV_ErrHandler:
    cpdef evaluate(self,
                   int error_code,
                   bytes module,
                   bytes function,
                   bytes msg,
                   object user_data = None):
        """ format that error handling functions must match """
        pass

cdef class CV_WrapErrHandler(CV_ErrHandler):
    cpdef set_err_handler(self, object err_handler):
        """
        set some (c/p)ython function as the error handler
        """
        nrarg = _get_num_args(err_handler)
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

cdef void _cv_err_handler_fn(
    int error_code, const char *module, const char *function, char *msg,
    void *eh_data
):
    """
    function with the signature of CVErrHandlerFn, that calls python error
    handler
    """
    aux_data = <CV_data> eh_data
    aux_data.err_handler.evaluate(error_code,
                                  module,
                                  function,
                                  msg,
                                  aux_data.err_user_data)

# Auxiliary data carrying runtime vales for the CVODE solver
cdef class CV_data:
    def __cinit__(self, N):
        self.parallel_implementation = False
        self.user_data = None
        self.err_user_data = None

        self.yy_tmp = np.empty(N, DTYPE)
        self.yp_tmp = np.empty(N, DTYPE)
        self.jac_tmp = None
        self.g_tmp = None
        self.r_tmp = np.empty(N, DTYPE)
        self.z_tmp = np.empty(N, DTYPE)

cdef class CVODE:

    def __cinit__(self, Rfn, **options):
        """
        Initialize the CVODE Solver and it's default values

        Input:
            Rfn     - right-hand-side function
            options - additional options for initialization, for the list
                      of supported options and their values see set_options()

        """

        default_values = {
            'implementation': 'serial',
            'lmm_type': 'BDF',
            'nonlinsolver': 'newton',
            'rtol': 1e-6, 'atol': 1e-12,
            'linsolver': 'dense',
            'lband': 0,'uband': 0,
            'maxl': 0,
            'precond_type': 'NONE',
            'tstop': 0.,
            'order': 0,
            'max_step_size': 0.,
            'min_step_size': 0.,
            'first_step_size': 0.,
            'max_steps': 0,
            'bdf_stability_detection': False,
            'max_conv_fails': 0,
            'max_nonlin_iters': 0,
            'nonlin_conv_coef': 0.,
            'one_step_compute': False,
            'user_data': None,
            'rfn': None,
            'rootfn': None,
            'nr_rootfns': 0,
            'jacfn': None,
            'prec_setupfn': None,
            'prec_solvefn': None,
            'jac_times_vecfn': None,
            'jac_times_setupfn': None,
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
        self._old_api = False # use new api by default
        self._step_compute = False #avoid dict lookup
        self._validate_flags = False # don't validate by default
        self.set_options(rfn=Rfn, **options)
        self._cv_mem = NULL
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
        for (key, value) in options.items():
            if key.lower() in self.options:
                self.options[key.lower()] = value
            else:
                raise ValueError("Option '%s' is not supported by solver" % key)

        # If the solver is running, this re-sets runtime changeable options,
        # otherwise it does nothing
        self._set_runtime_changeable_options(options)

    cpdef _set_runtime_changeable_options(self, object options,
                                          bint supress_supported_check=False):
        """
          (Re)Set options that can change after CV_MEM object was allocated -
          mostly useful for values that need to be changed at run-time.
        """
        cdef int flag
        cdef void* cv_mem = self._cv_mem

        # Don't do anything if we are not at runtime
        if cv_mem is NULL:
            return 0

        # Check whether options are all only supported - this check can
        # be supressed by 'supress_supported_check = True'
        if not supress_supported_check:
            for opt in options.keys():
                if not opt in ['atol', 'rtol', 'tstop', 'rootfn', 'nr_rootfns',
                               'verbosity', 'one_step_compute']:
                    raise ValueError("Option '%s' can''t be set runtime." % opt)

        # Verbosity level
        if ('verbosity' in options) and (options['verbosity'] is not None):
            verbosity = options['verbosity']
            self.options['verbosity'] = verbosity
            self.verbosity = verbosity
            warn("verbosity is deprecated, control output via logging and "
                 "err_handler", DeprecationWarning
            )

        # Root function (rootfn and nr_rootfns)
        if ('rootfn' in options) and (options['rootfn'] is not None):

            # Set root function to internal options...
            rootfn     = options['rootfn']
            nr_rootfns = options['nr_rootfns']

            if nr_rootfns is None:
                raise ValueError('Number of root-ing functions ''nr_rootfns'' '
                                 'must be specified.')

            if not isinstance(rootfn, CV_RootFunction):
                tmpfun = CV_WrapRootFunction()
                tmpfun.set_rootfn(rootfn)
                rootfn = tmpfun

            self.options['rootfn'] = rootfn
            self.options['nr_rootfns'] = nr_rootfns

            # ...and to the auxiliary data object (holding runtime data)
            self.aux_data.rootfn = rootfn
            self.aux_data.g_tmp  = np.empty([nr_rootfns,], DTYPE)

            # TODO: Shouldn't be the rootn in the cvode obj unset first?
            flag = CVodeRootInit(cv_mem, nr_rootfns, _rootfn)

            if flag == CV_SUCCESS:
                pass
            if flag == CV_ILL_INPUT:
                raise ValueError('CVRootInit: Function ''rootfn'' is NULL '
                                 'but ''nr_rootfns'' > 0')
            elif flag == CV_MEM_FAIL:
                raise MemoryError('CVRootInit: Memory allocation error')

        # Set atol and rtol tolerances
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
                flag = CVodeSStolerances(cv_mem, <realtype> opts_rtol,
                                                 <realtype> opts_atol)
            else:
                np_atol = np.asarray(opts_atol, dtype=DTYPE)
                if np.alen(np_atol) != self.N:
                    raise ValueError("Array length inconsistency: 'atol' "
                                     "lenght (%i) differs from problem size "
                                     "(%i)." % (np.alen(np_atol), self.N))

                if self.parallel_implementation:
                    raise NotImplemented
                else:
                    atol = N_VMake_Serial(self.N, <realtype *> np_atol.data)
                    flag = CVodeSVtolerances(cv_mem, <realtype> opts_rtol, atol)

                    self.atol = atol

            if flag == CV_ILL_INPUT:
                raise ValueError("CVode tolerances: negative 'atol' or 'rtol' value.")
            elif flag != CV_SUCCESS:
                raise ValueError("CVode tolerances: tolerance could not be set")
            #TODO: implement CVFWtolerances(cv_mem, efun)

        # Set tstop
        if ('tstop' in options) and (options['tstop'] is not None):
            opts_tstop = options['tstop']
            self.options['tstop'] = opts_tstop
            if (not opts_tstop is None) and (opts_tstop > 0.):
                flag = CVodeSetStopTime(cv_mem, <realtype> opts_tstop)
                if flag == CV_ILL_INPUT:
                    raise ValueError('CVodeSetStopTime::Stop value is beyond '
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
            if not isinstance(fn, CV_ContinuationFunction):
                fn = CV_ContinuationFunction(fn)
            self.options['onroot'] = fn
        elif self.options.get('onroot', None) is None:
            self.options['onroot'] = CV_ContinuationFunction(no_continue_fn)

        # Set ontstop
        if options.get('ontstop', None) is not None:
            fn = options['ontstop']
            if not isinstance(fn, CV_ContinuationFunction):
                fn = CV_ContinuationFunction(fn)
            self.options['ontstop'] = fn
        elif self.options.get('ontstop', None) is None:
            self.options['ontstop'] = CV_ContinuationFunction(no_continue_fn)

        # Set validate status
        if options.get('validate_flags') is not None:
            validate_flags = options["validate_flags"]
            if not validate_flags in [True, False]:
                raise ValueError("Option 'validate_flags' must be True or False")
            if validate_flags and self._old_api:
                raise ValueError("Option 'validate_flags' requires 'old_api' to be False")
            self.options['validate_flags'] = validate_flags
            self._validate_flags = options["validate_flags"]

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
        cdef np.ndarray[DTYPE_t, ndim=1] np_y0
        np_y0 = np.asarray(y0, dtype=DTYPE)

        #flag is always True, as errors are exceptions for cvode init_step!
        (flag, time) = self._init_step(t0, np_y0)

        if self._old_api:
            return (flag, time)
        else:
            y_retn  = np.empty(np.alen(np_y0), DTYPE)
            y_retn[:] = np_y0[:]
            soln = SolverReturn(
                flag=flag,
                values=SolverVariables(t=time, y=y_retn),
                errors=SolverVariables(t=None, y=None),
                roots=SolverVariables(t=None, y=None),
                tstop=SolverVariables(t=None, y=None),
                message=STATUS_MESSAGE[StatusEnum.SUCCESS]
            )
            if self._validate_flags:
                return self.validate_flags(soln)
            return soln

    cpdef _init_step(self, DTYPE_t t0,
                     np.ndarray[DTYPE_t, ndim=1] y0):

        cdef dict opts = self.options

        lmm_type  = opts['lmm_type'].lower()
        nonlinsolver = opts['nonlinsolver'].lower()

        if lmm_type == 'bdf':
            lmm = CV_BDF
        elif lmm_type == 'adams':
            lmm = CV_ADAMS
        else:
            raise ValueError('CVODE:init: Unrecognized lmm_type: %s' % lmm_type)

        if nonlinsolver not in ['fixedpoint', 'newton']:
            raise ValueError('CVODE:init: Unrecognized nonlinsolver: %s'
                             % nonlinsolver)

        self.parallel_implementation = \
          (opts['implementation'].lower() == 'parallel')
        if self.parallel_implementation:
            raise ValueError('Error: Parallel implementation not implemented !')
        cdef INDEX_TYPE_t N
        N = <INDEX_TYPE_t> np.alen(y0)

        if opts['rfn'] is None:
            raise ValueError('The right-hand-side function rfn not assigned '
                              'during ''set_options'' call !')

        #TODO: when implementing parallel, does N_VDestroy be called separately
        #      for parallel version or it's a generic one?
        if not self.y0 is NULL:
            N_VDestroy(self.y0)
            N_VDestroy(self.y)
            N_VDestroy(self.yp)

        # Initialize y0, y, yp
        if self.parallel_implementation:
            raise NotImplemented
        else:
            self.y0 = N_VMake_Serial(N, <realtype *>y0.data)
            self.y  = N_VClone(self.y0)
            self.yp = N_VNew_Serial(N)

        cdef int flag
        cdef void* cv_mem = self._cv_mem

        if (cv_mem is NULL) or (self.N != N):
            if (not cv_mem is NULL):
                CVodeFree(&cv_mem)
            cv_mem = CVodeCreate(lmm)
            if cv_mem is NULL:
                raise MemoryError('CVodeCreate:MemoryError: Could not create '
                                  'cv_mem object')

            self._cv_mem = cv_mem
            flag = CVodeInit(cv_mem, _rhsfn,  <realtype> t0, self.y0)
        elif self.N == N:
            flag = CVodeReInit(cv_mem, <realtype> t0, self.y0)
        else:
            raise ValueError('CVodeInit:Error: You should not be here...')
        if flag == CV_ILL_INPUT:
                raise ValueError('CVode[Re]Init: Ill input')
        elif flag == CV_MEM_FAIL:
            raise MemoryError('CV[Re]Init: Memory allocation error')
        elif flag == CV_MEM_NULL:
            raise MemoryError('CVodeCreate: Memory allocation error')
        elif flag == CV_NO_MALLOC:
            raise MemoryError('CVodeReInit: No memory allocated in CVInit.')

        self.N = N

        # Initialize auxiliary variables
        self.aux_data = CV_data(N)

        # Set err_handler
        err_handler = opts.get('err_handler', None)
        if err_handler is not None:
            if not isinstance(err_handler, CV_ErrHandler):
                tmpfun = CV_WrapErrHandler()
                tmpfun.set_err_handler(err_handler)
                err_handler = tmpfun

            self.aux_data.err_handler = err_handler

            flag = CVodeSetErrHandlerFn(
                cv_mem, _cv_err_handler_fn, <void*> self.aux_data)

            if flag == CV_SUCCESS:
                pass
            elif flag == CV_MEM_FAIL:
                raise MemoryError(
                    'CVodeSetErrHandlerFn: Memory allocation error')
            else:
                raise RuntimeError('CVodeSetErrHandlerFn: Unknown flag raised')
        self.aux_data.err_user_data = opts['err_user_data'] or opts['user_data']

        self.aux_data.parallel_implementation = self.parallel_implementation

        rfn = opts['rfn']
        if not isinstance(rfn , CV_RhsFunction):
            tmpfun = CV_WrapRhsFunction()
            tmpfun.set_rhsfn(rfn)
            rfn = tmpfun
            opts['rfn'] = tmpfun
        self.aux_data.rfn = rfn

        jac = opts['jacfn']
        if jac is not None and not isinstance(jac , CV_JacRhsFunction):
            tmpfun = CV_WrapJacRhsFunction()
            tmpfun.set_jacfn(jac)
            jac = tmpfun
            opts['jacfn'] = tmpfun
        self.aux_data.jac = jac

        #we test if rfn call doesn't give errors due to bad coding, as
        #cvode will ignore errors, it only checks return value (0 or 1 for error)
        if isinstance(rfn, CV_WrapRhsFunction):
            _test = np.empty(np.alen(y0), DTYPE)
            if rfn.with_userdata:
                rfn._rhsfn(t0, y0, _test, opts['user_data'])
            else:
                rfn._rhsfn(t0, y0, _test)
            _test = None

        prec_setupfn = opts['prec_setupfn']
        if prec_setupfn is not None and not isinstance(prec_setupfn, CV_PrecSetupFunction):
            tmpfun = CV_WrapPrecSetupFunction()
            tmpfun.set_prec_setupfn(prec_setupfn)
            prec_setupfn = tmpfun
            opts['prec_setupfn'] = tmpfun
        self.aux_data.prec_setupfn = prec_setupfn

        prec_solvefn = opts['prec_solvefn']
        if prec_solvefn is not None and not isinstance(prec_solvefn, CV_PrecSolveFunction):
            tmpfun = CV_WrapPrecSolveFunction()
            tmpfun.set_prec_solvefn(prec_solvefn)
            prec_solvefn = tmpfun
            opts['prec_solvefn'] = tmpfun
        self.aux_data.prec_solvefn = prec_solvefn

        jac_times_vecfn = opts['jac_times_vecfn']
        if jac_times_vecfn is not None and not isinstance(jac_times_vecfn, CV_JacTimesVecFunction):
            tmpfun = CV_WrapJacTimesVecFunction()
            tmpfun.set_jac_times_vecfn(jac_times_vecfn)
            jac_times_vecfn = tmpfun
            opts['jac_times_vecfn'] = tmpfun
        self.aux_data.jac_times_vecfn = jac_times_vecfn

        jac_times_setupfn = opts['jac_times_setupfn']
        if jac_times_setupfn is not None and not isinstance(jac_times_setupfn, CV_JacTimesSetupFunction):
            tmpfun = CV_WrapJacTimesSetupFunction()
            tmpfun.set_jac_times_setupfn(jac_times_setupfn)
            jac_times_setupfn = tmpfun
            opts['jac_times_setupfn'] = tmpfun
        self.aux_data.jac_times_setupfn = jac_times_setupfn

        self.aux_data.user_data = opts['user_data']

        # As cv_mem is now initialized, set also options changeable at runtime
        self._set_runtime_changeable_options(opts, supress_supported_check=True)

        # As user data we pass the CV_data object
        CVodeSetUserData(cv_mem, <void*> self.aux_data)

        if (opts['order'] > 0):
            CVodeSetMaxOrd(cv_mem, <int> opts['order'])
        CVodeSetMaxNumSteps(cv_mem, <int> opts['max_steps'])
        if lmm_type == 'bdf':
            CVodeSetStabLimDet(cv_mem, <bint> opts['bdf_stability_detection'])
        if opts['first_step_size'] > 0.:
            CVodeSetInitStep(cv_mem, <realtype> opts['first_step_size'])
        if (opts['min_step_size'] > 0.):
           CVodeSetMinStep(cv_mem, <realtype> opts['min_step_size'])
        if opts['max_step_size'] > 0.:
            flag = CVodeSetMaxStep(cv_mem, <realtype> opts['max_step_size'])
            if flag == CV_ILL_INPUT:
                raise ValueError('CVodeSetMaxStep: max_step_size is negative '
                                 'or smaller than min_step_size.')
        if opts['max_nonlin_iters'] > 0:
            CVodeSetMaxNonlinIters(cv_mem, <int> opts['max_nonlin_iters'])
        if opts['max_conv_fails'] > 0:
            CVodeSetMaxConvFails(cv_mem, <int> opts['max_conv_fails'])
        if opts['nonlin_conv_coef'] > 0:
            CVodeSetNonlinConvCoef(cv_mem, <int> opts['nonlin_conv_coef'])

        # Linsolver
        linsolver = opts['linsolver'].lower()
        if nonlinsolver == 'newton':
            if linsolver == 'dense':
                A = SUNDenseMatrix(N, N)
                LS = SUNDenseLinearSolver(self.y0, A)
                # check if memory was allocated
                if (A == NULL or LS == NULL):
                    raise ValueError('Could not allocate matrix or linear solver')
                # attach matrix and linear solver to cvode
                flag = CVodeSetLinearSolver(cv_mem, LS, A)
                if flag == CVLS_ILL_INPUT:
                    raise ValueError('CVDense linear solver setting failed, '
                                    'arguments incompatible')
                elif flag == CVLS_MEM_FAIL:
                    raise MemoryError('CVDense linear solver memory allocation error.')
                elif flag != CVLS_SUCCESS:
                    raise ValueError('CVodeSetLinearSolver failed with code {}'
                                     .format(flag))
            elif linsolver == 'band':
                A = SUNBandMatrix(N, <int> opts['uband'], <int> opts['lband']);
                LS = SUNBandLinearSolver(self.y0, A);
                if (A == NULL or LS == NULL):
                    raise ValueError('Could not allocate matrix or linear solver')
                flag = CVodeSetLinearSolver(cv_mem, LS, A)
    
                if flag == CVLS_ILL_INPUT:
                    raise ValueError('CVBand linear solver  setting failed, '
                                     'arguments incompatible')
                elif flag == CVLS_MEM_FAIL:
                    raise MemoryError('CVBand linear solver memory allocation error.')
                elif flag != CVLS_SUCCESS:
                    raise ValueError('CVodeSetLinearSolver failed with code {}'
                                     .format(flag))
            elif linsolver == 'diag':
                flag = CVDiag(cv_mem)
                if flag == CVDIAG_ILL_INPUT:
                        raise ValueError('CVDiag solver is not compatible with'
                                         ' the current nvector implementation.')
                elif flag == CVDIAG_MEM_FAIL:
                        raise MemoryError('CVDiag memory allocation error.')
                elif flag != CVDIAG_SUCCESS:
                    raise ValueError('CVDiag failed with code {}'
                                     .format(flag))
            elif ((linsolver == 'spgmr') or (linsolver == 'spbcgs')
                  or (linsolver == 'sptfqmr')):
                precond_type = opts['precond_type'].lower()
                if precond_type == 'none':
                    pretype = PREC_NONE
                elif precond_type == 'left':
                    pretype = PREC_LEFT
                elif precond_type == 'right':
                    pretype = PREC_RIGHT
                elif precond_type == 'both':
                    pretype = PREC_BOTH
                else:
                    raise ValueError('LinSolver::Precondition: Unknown type: %s'
                                     % opts['precond_type'])
    
                if linsolver == 'spgmr':
                    LS = SUNSPGMR(self.y0, pretype, <int> opts['maxl']);
                    if LS == NULL:
                        raise ValueError('Could not allocate linear solver')
                elif linsolver == 'spbcgs':
                    LS = SUNSPBCGS(self.y0, pretype, <int> opts['maxl']);
                    if LS == NULL:
                        raise ValueError('Could not allocate linear solver')
                elif linsolver == 'sptfqmr':
                    LS = SUNSPTFQMR(self.y0, pretype, <int> opts['maxl']);
                    if LS == NULL:
                        raise ValueError('Could not allocate linear solver')
                else:
                    raise ValueError('Given linsolver {} not implemented in odes'.format(linsolver))
                    
                flag = CVodeSetLinearSolver(cv_mem, LS, NULL);
                if flag == CVLS_MEM_FAIL:
                        raise MemoryError('LinSolver:CVSpils memory allocation '
                                          'error.')
                elif flag != CVLS_SUCCESS:
                    raise ValueError('CVodeSetLinearSolver failed with code {}'
                                     .format(flag))
                # TODO: make option for the Gram-Schmidt orthogonalization
                #flag = SUNSPGMRSetGSType(LS, gstype);
                                          
                # TODO make option
                #flag = CVSpilsSetEpsLin(cvode_mem, DELT);
                if self.aux_data.prec_solvefn:
                    if self.aux_data.prec_setupfn:
                        flag = CVodeSetPreconditioner(cv_mem, _prec_setupfn, _prec_solvefn)
                    else:
                        flag = CVodeSetPreconditioner(cv_mem, NULL, _prec_solvefn)
                if flag == CVLS_MEM_NULL:
                    raise ValueError('LinSolver: The cvode mem pointer is NULL.')
                elif flag == CVLS_LMEM_NULL:
                    raise ValueError('LinSolver: The cvspils linear solver has '
                                     'not been initialized.')
                elif flag != CVLS_SUCCESS:
                    raise ValueError('CVodeSetPreconditioner failed with code {}'
                                     .format(flag))
    
                if self.aux_data.jac_times_vecfn:
                    if self.aux_data.jac_times_setupfn:
                       flag = CVodeSetJacTimes(cv_mem, _jac_times_setupfn, 
                                               _jac_times_vecfn)
                    else:
                       flag = CVodeSetJacTimes(cv_mem, NULL, _jac_times_vecfn)
                if flag == CVLS_MEM_NULL:
                    raise ValueError('LinSolver: The cvode mem pointer is NULL.')
                elif flag == CVLS_LMEM_NULL:
                    raise ValueError('LinSolver: The cvspils linear solver has '
                                     'not been initialized.')
                elif flag != CVLS_SUCCESS:
                    raise ValueError('CVSpilsSetJacTimes failed with code {}'
                                     .format(flag))
            else:
                if SUNDIALS_BLAS_LAPACK:
                    if linsolver == 'lapackdense':
                        A = SUNDenseMatrix(N, N)
                        LS = SUNLapackDense(self.y0, A)
                        # check if memory was allocated
                        if (A == NULL or LS == NULL):
                            raise ValueError('Could not allocate matrix or linear solver')
                        # attach matrix and linear solver to cvode
                        flag = CVodeSetLinearSolver(cv_mem, LS, A)
                        if flag == CVLS_ILL_INPUT:
                            raise ValueError('CVDense lapack linear solver setting failed, '
                                            'arguments incompatible')
                        elif flag == CVLS_MEM_FAIL:
                            raise MemoryError('CVDense lapack linear solver memory allocation error.')
                        elif flag != CVLS_SUCCESS:
                            raise ValueError('CVodeSetLinearSolver failed with code {}'
                                             .format(flag))
                    elif linsolver == 'lapackband':
                        A = SUNBandMatrix(N, <int> opts['uband'], <int> opts['lband'])
                        LS = SUNLapackBand(self.y0, A)
                        if (A == NULL or LS == NULL):
                            raise ValueError('Could not allocate matrix or linear solver')
                        flag = CVodeSetLinearSolver(cv_mem, LS, A)
                        if flag == CVLS_ILL_INPUT:
                            raise ValueError('CVLapackBand linear solver setting failed, '
                                             'arguments incompatible')
                        elif flag == CVLS_MEM_FAIL:
                            raise MemoryError('CVLapackBand linear solver memory allocation error.')
                        elif flag != CVLS_SUCCESS:
                            raise ValueError('CVodeSetLinearSolver failed with code {}'
                                             .format(flag))
                    else:
                        raise ValueError('LinSolver: Unknown solver type: %s'
                                             % opts['linsolver'])
                elif linsolver in ['lapackdense', 'lapackband']:
                    raise ValueError('LinSolver: LAPACK not available, cannot execute solver type: %s'
                                         % opts['linsolver'])
                else:
                    raise ValueError('LinSolver: Unknown solver type: %s'
                                         % opts['linsolver'])
        elif nonlinsolver == 'fixedpoint':
            # create fixed point nonlinear solver object 
            NLS = SUNNonlinSol_FixedPoint(self.y0, 0);
            # attach nonlinear solver object to CVode
            flag = CVodeSetNonlinearSolver(cv_mem, NLS)
            if flag != CV_SUCCESS:
                raise ValueError('CVodeSetNonlinearSolver failed with code {}'
                                 .format(flag))
        
        if (linsolver in ['dense', 'lapackdense', 'lapackband', 'band']
            and self.aux_data.jac):
            # we need to create the correct shape for jacobian output, here is
            # the best place
            if linsolver == 'lapackband' or linsolver == 'band':
                self.aux_data.jac_tmp = np.empty((
                        opts['uband'] + opts['lband'] + 1,
                        np.alen(y0)
                    ), DTYPE
                )
            else:
                self.aux_data.jac_tmp = np.empty((np.alen(y0), np.alen(y0)), DTYPE)
            CVDlsSetJacFn(cv_mem, _jacdense)

        #we test if jac don't give errors due to bad coding, as
        #cvode will ignore errors, it only checks return value (0 or 1 for error)
        if jac is not None and isinstance(jac, CV_WrapJacRhsFunction):
            if linsolver == 'lapackband' or linsolver == 'band':
                _test = np.empty((opts['uband']+opts['lband']+1, np.alen(y0)),
                        DTYPE)
            else:
                _test = np.empty((np.alen(y0), np.alen(y0)), DTYPE)
            _fy_test = np.zeros(np.alen(y0), DTYPE)
            jac._jacfn(t0, y0, _fy_test, _test)
            _test = None

        #now we initialize storage which is persistent over steps
        self.t_roots = []
        self.y_roots = []
        self.t_tstop = []
        self.y_tstop = []

        #TODO: Reinitialization of the rooting function

        self.initialized = True

        return (True, t0)

    def reinit_IC(self, DTYPE_t t0, object y0):
        """
        Re-initialize (only) the initial condition IC without re-setting also
        all the remaining solver options. See also 'init_step()' funtion.
        Typically, an exception is raised with wrong input.

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

        """

        cdef np.ndarray[DTYPE_t, ndim=1] np_y0
        np_y0 = np.asarray(y0)

        flag, time = self._reinit_IC(t0, np_y0)

        if self._old_api:
            return (flag, time)
        else:
            y_retn  = np.empty(np.alen(np_y0), DTYPE)
            y_retn[:] = np_y0[:]
            soln = SolverReturn(
                flag=flag,
                values=SolverVariables(t=time, y=y_retn),
                errors=SolverVariables(t=None, y=None),
                roots=SolverVariables(t=None, y=None),
                tstop=SolverVariables(t=None, y=None),
                message=STATUS_MESSAGE[StatusEnum.SUCCESS]
            )
            if self._validate_flags:
                return self.validate_flags(soln)
            return soln

    cpdef _reinit_IC(self, DTYPE_t t0, np.ndarray[DTYPE_t, ndim=1] y0):
        # If not yet initialized, run full initialization
        if self.y0 is NULL:
            self._init_step(t0, y0)
            return

        cdef INDEX_TYPE_t N
        N = <INDEX_TYPE_t> np.alen(y0)
        if N == self.N:
            self.y0  = N_VMake_Serial(N, <realtype *>y0.data)
        else:
            raise ValueError("Cannot re-init IC with array of unequal lenght.")

        flag = CVodeReInit(self._cv_mem, <realtype> t0, self.y0)

        if flag == CV_ILL_INPUT:
            raise ValueError('CVodeReInit: Ill input')
        elif flag == CV_MEM_FAIL:
            raise MemoryError('CVReInit: Memory allocation error')
        elif flag == CV_MEM_NULL:
            raise MemoryError('CVodeReCreate: Memory allocation error')
        elif flag == CV_NO_MALLOC:
            raise MemoryError('CVodeReInit: No memory allocated in CVInit.')

        return (True, t0)

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
        cdef np.ndarray[DTYPE_t, ndim=1] np_tspan, np_y0

        if not np.alen(tspan) > 1:
            raise ValueError("Solve tspan must be array with minimum 2 elements,"
                             " start and end time.")
        np_tspan = np.asarray(tspan, dtype=DTYPE)
        np_y0    = np.asarray(y0, dtype=DTYPE)


        soln = self._solve(np_tspan, np_y0)
        if self._old_api:
            warn("Old api is deprecated, move to new api", DeprecationWarning)
        flag = StatusEnum(soln[0])
        t, y, t_err, y_err = soln[1:5]

        t_roots = np.array(soln[5]) if soln[5] else None
        y_roots = np.array(soln[6]) if soln[6] else None
        t_tstop = np.array(soln[7]) if soln[7] else None
        y_tstop = np.array(soln[8]) if soln[8] else None

        if self._old_api:
            if flag == StatusEnum.ROOT_RETURN:
                return flag, t, y, t_roots[0], y_roots[0]
            elif flag == StatusEnum.TSTOP_RETURN:
                return flag, t, y, t_tstop[0], y_tstop[0]
            return flag, t, y, t_err, y_err

        soln = SolverReturn(
            flag=flag,
            values=SolverVariables(t=t, y=y),
            errors=SolverVariables(t=t_err, y=y_err),
            roots=SolverVariables(t=t_roots, y=y_roots),
            tstop=SolverVariables(t=t_tstop, y=y_tstop),
            message=STATUS_MESSAGE[flag]
        )
        if self._validate_flags:
            return self.validate_flags(soln)
        return soln

    cpdef _solve(self, np.ndarray[DTYPE_t, ndim=1] tspan,
                 np.ndarray[DTYPE_t, ndim=1] y0):

        cdef np.ndarray[DTYPE_t, ndim=1] t_retn
        cdef np.ndarray[DTYPE_t, ndim=2] y_retn
        t_retn  = np.empty(np.shape(tspan), DTYPE)
        y_retn  = np.empty([np.alen(tspan), np.alen(y0)], DTYPE)

        self._init_step(tspan[0], y0)
        PyErr_CheckSignals()
        t_retn[0] = tspan[0]
        y_retn[0, :] = y0

        cdef np.ndarray[DTYPE_t, ndim=1] y_last
        cdef unsigned int idx = 1 # idx == 0 is IC
        cdef unsigned int last_idx = np.alen(tspan)
        cdef DTYPE_t t
        cdef int flag = 0
        cdef void *cv_mem = self._cv_mem
        cdef realtype t_out
        cdef N_Vector y  = self.y
        cdef CV_ContinuationFunction onroot = self.options['onroot']
        cdef CV_ContinuationFunction ontstop = self.options['ontstop']
        cdef object y_err
        cdef object t_err

        y_last   = np.empty(np.shape(y0), DTYPE)
        t = tspan[idx]

        while True:
            flag = CVode(cv_mem, <realtype> t,  y, &t_out, CV_NORMAL)

            nv_s2ndarray(y,  y_last)

            if flag == CV_SUCCESS or flag == CV_WARNING:
                t_retn[idx]    = t_out
                y_retn[idx, :] = y_last
                idx = idx + 1
                PyErr_CheckSignals()

                # Iterate until we reach the end of tspan
                if idx < last_idx:
                    t = tspan[idx]
                    continue
                else:
                    break

            elif flag == CV_ROOT_RETURN:
                self.t_roots.append(np.copy(t_out))
                self.y_roots.append(np.copy(y_last))
                root_flag = onroot.evaluate(t_out, y_last, self)
                if (t_out == t):
                    #a root at our wanted output, and we continue comp
                    t_retn[idx]    = t_out
                    y_retn[idx, :] = y_last
                    idx = idx + 1
                    if idx < last_idx:
                        t = tspan[idx]
                    else:
                        break
                if root_flag == 0:
                    PyErr_CheckSignals()
                    continue
                break
            elif flag == CV_TSTOP_RETURN:
                self.t_tstop.append(np.copy(t_out))
                self.y_tstop.append(np.copy(y_last))
                tstop_flag = ontstop.evaluate(t_out, y_last, self)
                if (t_out == t):
                    #a tstop at our wanted output, and we continue comp
                    t_retn[idx]    = t_out
                    y_retn[idx, :] = y_last
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
        if flag < 0:
            y_err = y_last
            t_err = t_out
        else:
            y_err = None
            t_err = None
        return (
            flag, t_retn, y_retn, t_err, y_err, self.t_roots, self.y_roots,
            self.t_tstop, self.y_tstop,
        )


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
        if not self.initialized:
            raise ValueError('Method ''init_step'' has to be called prior to '
                             'the first call of ''step'' method.')

        cdef N_Vector y  = self.y
        cdef realtype t_out
        cdef int flagCV

        if self._old_api:
            if t>0.0:
                flagCV = CVode(self._cv_mem, <realtype> t,  y, &t_out, CV_NORMAL)
            else:
                flagCV = CVode(self._cv_mem, <realtype> -t,  y, &t_out, CV_ONE_STEP)
        else:
            if self._step_compute:
                flagCV = CVode(self._cv_mem, <realtype> t,  y, &t_out, CV_ONE_STEP)
            else:
                flagCV = CVode(self._cv_mem, <realtype> t,  y, &t_out, CV_NORMAL)

        if self._old_api:
            warn("Old api is deprecated, move to new api", DeprecationWarning)

        cdef np.ndarray[DTYPE_t, ndim=1] y_out, y_err
        if not y_retn is None:
            nv_s2ndarray(y, y_retn)
            y_out = y_retn
        else:
            y_out  = np.empty(self.N, DTYPE)
            nv_s2ndarray(y, y_out)

        flag = StatusEnum(flagCV)

        t_err = None
        y_err = None
        sol_t_out = t_out
        if flagCV == CV_SUCCESS or flag == CV_WARNING:
            pass
        elif flagCV == CV_ROOT_RETURN:
            self.t_roots.append(np.copy(t_out))
            self.y_roots.append(np.copy(y_out))
        elif flagCV == CV_TSTOP_RETURN:
            self.t_tstop.append(np.copy(t_out))
            self.y_tstop.append(np.copy(y_out))
        elif flagCV < 0:
            t_err = np.copy(t_out)
            y_err = np.copy(y_out)
            sol_t_out = None
            y_out = None

        PyErr_CheckSignals()

        t_roots = np.array(self.t_roots) if self.t_roots else None
        y_roots = np.array(self.y_roots) if self.y_roots else None
        t_tstop = np.array(self.t_tstop) if self.t_tstop else None
        y_tstop = np.array(self.y_tstop) if self.y_tstop else None

        if self._old_api:
            return flagCV, t_out

        return SolverReturn(
            flag=flag,
            values=SolverVariables(t=sol_t_out, y=y_out),
            errors=SolverVariables(t=t_err, y=y_err),
            roots=SolverVariables(t=t_roots, y=y_roots),
            tstop=SolverVariables(t=t_tstop, y=y_tstop),
            message=STATUS_MESSAGE[flag]
        )

    def validate_flags(self, soln):
        """
        Validates the flag returned by `CVODE.solve`.

        Validation happens using the following scheme:
         * failures (`flag` < 0) raise `CVODESolveFailed` or a subclass of it;
         * finding a root (and stopping) raises `CVODESolveFoundRoot`;
         * reaching `tstop` (and stopping) raises `CVODESolveReachedTSTOP`;
         * otherwise, return an instance of `SolverReturn`.

        In the case where ontstop or onroot are used, `CVODESolveFoundRoot` or
        `CVODESolveReachedTSTOP` will be raised only if the solver is told to
        stop at that point.
        """
        if soln.flag == StatusEnum.SUCCESS:
            return soln
        if soln.flag < 0:
            raise CVODESolveFailed(soln)
        elif soln.flag == StatusEnum.TSTOP_RETURN:
            raise CVODESolveReachedTSTOP(soln)
        elif soln.flag == StatusEnum.ROOT_RETURN:
            raise CVODESolveFoundRoot(soln)
        warn(WARNING_STR.format(soln.flag, *soln.err_values))
        return soln

    def get_info(self):
        """
        Returns a dict with information about the solver, like number
        of calls to the user's right-hand side function.

        """
        cdef int flagCV
        cdef long int nsteps, nfevals, nlinsetups, netfails
        cdef int qlast, qcur
        cdef realtype hinused, hlast, hcur, tcur

        # for extra output from SPILS modules (SPGMR, SPBCGS, SPTFQMR)
        cdef long int npevals, npsolves, njvevals, nliters, nfevalsLS

        flagCV = CVodeGetIntegratorStats(self._cv_mem, &nsteps, &nfevals,
                                         &nlinsetups, &netfails, &qlast, &qcur,
                                         &hinused, &hlast, &hcur, &tcur)
        # following the names of the CVodeGet* optional output functions
        # c.f. Table 4.2 of User Documentation for CVODE
        # http://computation.llnl.gov/sites/default/files/public/cv_guide.pdf
        info = {'NumSteps': nsteps, 'NumRhsEvals': nfevals,
                'NumLinSolvSetups': nlinsetups, 'NumErrTestFails': netfails,
                'LastOrder': qlast, 'CurrentOrder': qcur,
                'ActualInitStep': hinused,
                'LastStep': hlast, 'CurrentStep': hcur, 'CurrentStep': tcur}

        linsolver = self.options['linsolver'].lower()
        if linsolver == 'spgmr' or linsolver == 'spbcgs' or linsolver == 'sptfqmr':
            flagCV = CVSpilsGetNumPrecEvals(self._cv_mem, &npevals)
            flagCV = CVSpilsGetNumPrecSolves(self._cv_mem, &npsolves)
            flagCV = CVSpilsGetNumJtimesEvals(self._cv_mem, &njvevals)
            flagCV = CVSpilsGetNumLinIters(self._cv_mem, &nliters)
            flagCV = CVSpilsGetNumRhsEvals(self._cv_mem, &nfevalsLS)
            info.update({'NumPrecEvals': npevals, 'NumPrecSolves': npsolves,
                         'NumJtimesEvals': njvevals, 'NumLinIters': nliters,
                         'NumRhsEvalsJtimesFD': nfevalsLS})

        return info

    def __dealloc__(self):
        if not self._cv_mem is NULL: CVodeFree(&self._cv_mem)
        #TODO: when implementing parallel, does N_VDestroy be called separately
        #      for parallel version or it's a generic one?
        if not self.y0   is NULL: N_VDestroy(self.y0)
        if not self.y    is NULL: N_VDestroy(self.y)
        if not self.atol is NULL: N_VDestroy(self.atol)
