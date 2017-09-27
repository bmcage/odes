cimport numpy as np
from .c_sundials cimport N_Vector, realtype
from .common_defs cimport DTYPE_t

cdef class IDA_RhsFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       np.ndarray[DTYPE_t, ndim=1] result,
                       object userdata = *) except? -1

cdef class IDA_WrapRhsFunction(IDA_RhsFunction):
    cdef object _resfn
    cdef int with_userdata
    cpdef set_resfn(self, object resfn)

cdef class IDA_RootFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       np.ndarray[DTYPE_t, ndim=1] g,
                       object userdata = *) except? -1

cdef class IDA_WrapRootFunction(IDA_RootFunction):
    cpdef object _rootfn
    cdef int with_userdata
    cpdef set_rootfn(self, object rootfn)

cdef class IDA_JacRhsFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       DTYPE_t cj,
                       np.ndarray[DTYPE_t, ndim=2] J) except? -1

cdef class IDA_WrapJacRhsFunction(IDA_JacRhsFunction):
    cpdef object _jacfn
    cdef int with_userdata
    cpdef set_jacfn(self, object jacfn)

cdef class IDA_ContinuationFunction:
    cpdef object _fn
    cpdef int evaluate(self, DTYPE_t t, np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       IDA solver)

cdef int _res(realtype tt, N_Vector yy, N_Vector yp, N_Vector rr, void *self_obj)

cdef class IDA_ErrHandler:
    cpdef evaluate(self,
                   int error_code,
                   bytes module,
                   bytes function,
                   bytes msg,
                   object user_data = *)

cdef class IDA_WrapErrHandler(IDA_ErrHandler):
    cpdef object _err_handler
    cdef int with_userdata
    cpdef set_err_handler(self, object err_handler)


cdef class IDA_data:
    cdef np.ndarray yy_tmp, yp_tmp, residual_tmp, jac_tmp, g_tmp
    cdef IDA_RhsFunction res
    cdef IDA_JacRhsFunction jac
    cdef IDA_RootFunction rootfn
    cdef bint parallel_implementation
    cdef object user_data
    cdef IDA_ErrHandler err_handler
    cdef object err_user_data

cdef class IDA:
    cdef N_Vector atol

    cdef N_Vector y0, yp0, residual, y, yp
    cdef N_Vector dae_vars_id, constraints
    cdef long int N #problem size, i.e. len(y0) = N
    cdef list t_roots
    cdef list y_roots
    cdef list yp_roots
    cdef list t_tstop
    cdef list y_tstop
    cdef list yp_tstop

    cdef int order, nsteps
    cdef DTYPE_t maxstep, first_step
    cdef exclude_algvar_from_error, out
    cdef int compute_initcond
    cdef DTYPE_t compute_initcond_t0
    cdef long int mupper, mlower
    # ??? lband, uband, tcrit
    # ??? constraint_type, algebraic_var
    cdef bint initialized

    cdef void* _ida_mem
    cdef dict options
    cdef bint parallel_implementation
    cdef bint _old_api, _step_compute, _validate_flags
    cdef realtype t, t0

    cdef IDA_data aux_data

    cdef int verbosity

    #cdef realtype *y0, *yprime0

    # Functions

    #def set_options(self, dict options)
    cpdef _init_step(self, DTYPE_t t0,
                    np.ndarray[DTYPE_t, ndim=1] y0,
                    np.ndarray[DTYPE_t, ndim=1] yp0,
                    np.ndarray y_ic0_retn = ?,
                    np.ndarray yp_ic0_retn = ?)
    cpdef _reinit_IC(self, DTYPE_t t0,
                     np.ndarray[DTYPE_t, ndim=1] y0,
                     np.ndarray[DTYPE_t, ndim=1] yp0)
    cpdef _solve(self, np.ndarray[DTYPE_t, ndim=1] tspan,
                       np.ndarray[DTYPE_t, ndim=1] y0,
                       np.ndarray[DTYPE_t, ndim=1] yp0)
    cpdef _set_runtime_changeable_options(self, object options,
                                          bint supress_supported_check=*)
    #def step(self, realtype t)
