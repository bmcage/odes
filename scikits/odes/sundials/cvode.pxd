#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

cimport numpy as np
from .c_sundials cimport N_Vector, realtype
from .common_defs cimport DTYPE_t, INDEX_TYPE_t

cdef class CV_RhsFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       object userdata = *) except? -1

cdef class CV_WrapRhsFunction(CV_RhsFunction):
    cpdef public object _rhsfn
    cdef public int with_userdata
    cpdef set_rhsfn(self, object rhsfn)

cdef class CV_RootFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] g,
                       object userdata = *) except? -1

cdef class CV_WrapRootFunction(CV_RootFunction):
    cpdef object _rootfn
    cdef int with_userdata
    cpdef set_rootfn(self, object rootfn)

cdef class CV_JacRhsFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] fy,
                       np.ndarray[DTYPE_t, ndim=2] J) except? -1

cdef class CV_WrapJacRhsFunction(CV_JacRhsFunction):
    cpdef public object _jacfn
    cdef int with_userdata
    cpdef set_jacfn(self, object jacfn)

cdef class CV_PrecSetupFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       bint jok,
                       object jcurPtr,
                       DTYPE_t gamma,
                       object userdata = *) except? -1

cdef class CV_WrapPrecSetupFunction(CV_PrecSetupFunction):
    cpdef object _prec_setupfn
    cdef int with_userdata
    cpdef set_prec_setupfn(self, object prec_setupfn)

cdef class CV_PrecSolveFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] r,
                       np.ndarray[DTYPE_t, ndim=1] z,
                       DTYPE_t gamma,
                       DTYPE_t delta,
                       int lr,
                       object userdata = *) except? -1

cdef class CV_WrapPrecSolveFunction(CV_PrecSolveFunction):
    cpdef object _prec_solvefn
    cdef int with_userdata
    cpdef set_prec_solvefn(self, object prec_solvefn)


cdef class CV_JacTimesVecFunction:
    cpdef int evaluate(self,
                       np.ndarray[DTYPE_t, ndim=1] v,
                       np.ndarray[DTYPE_t, ndim=1] Jv,
                       DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       object userdata = *) except? -1

cdef class CV_WrapJacTimesVecFunction(CV_JacTimesVecFunction):
    cpdef object _jac_times_vecfn
    cdef int with_userdata
    cpdef set_jac_times_vecfn(self, object jac_times_vecfn)

cdef class CV_JacTimesSetupFunction:
    cpdef int evaluate(self,
                       DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] fy,
                       object userdata = *) except? -1

cdef class CV_WrapJacTimesSetupFunction(CV_JacTimesSetupFunction):
    cpdef object _jac_times_setupfn
    cdef int with_userdata
    cpdef set_jac_times_setupfn(self, object jac_times_setupfn)

cdef class CV_ContinuationFunction:
    cpdef object _fn
    cpdef int evaluate(self, DTYPE_t t, np.ndarray[DTYPE_t, ndim=1] y,
                       CVODE solver)

cdef class CV_ErrHandler:
    cpdef evaluate(self,
                   int error_code,
                   bytes module,
                   bytes function,
                   bytes msg,
                   object user_data = *)

cdef class CV_WrapErrHandler(CV_ErrHandler):
    cpdef object _err_handler
    cdef int with_userdata
    cpdef set_err_handler(self, object err_handler)


cdef class CV_data:
    cdef np.ndarray yy_tmp, yp_tmp, jac_tmp, g_tmp, r_tmp, z_tmp
    cdef CV_RhsFunction rfn
    cdef CV_JacRhsFunction jac
    cdef CV_RootFunction rootfn
    cdef CV_PrecSolveFunction prec_solvefn
    cdef CV_PrecSetupFunction prec_setupfn
    cdef CV_JacTimesVecFunction jac_times_vecfn
    cdef CV_JacTimesSetupFunction jac_times_setupfn
    cdef bint parallel_implementation
    cdef object user_data
    cdef CV_ErrHandler err_handler
    cdef object err_user_data

cdef class CVODE:
    cdef N_Vector atol
    cdef void* _cv_mem
    cdef dict options
    cdef bint parallel_implementation, initialized, _old_api, _step_compute, _validate_flags
    cdef CV_data aux_data

    cdef INDEX_TYPE_t N #problem size, i.e. len(y0) = N
    cdef N_Vector y0, y, yp # for 'step' method
    cdef list t_roots
    cdef list y_roots
    cdef list t_tstop
    cdef list y_tstop

    cdef int verbosity

    # Functions
    cpdef _init_step(self, DTYPE_t t0, np.ndarray[DTYPE_t, ndim=1] y0)
    cpdef _reinit_IC(self, DTYPE_t t0, np.ndarray[DTYPE_t, ndim=1] y0)

    cpdef _solve(self, np.ndarray[DTYPE_t, ndim=1] tspan,
                       np.ndarray[DTYPE_t, ndim=1] y0)
    cpdef _set_runtime_changeable_options(self, object options,
                                          bint supress_supported_check=*)
