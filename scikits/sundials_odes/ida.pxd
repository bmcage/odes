cimport numpy as np
from c_sundials cimport N_Vector, realtype
from common_defs cimport ResFunction, JacFunction

ctypedef np.float_t DTYPE_t

cdef int _res(realtype tt, N_Vector yy, N_Vector yp, N_Vector rr, void *self_obj)

cdef class IDA:
    cdef N_Vector atol
    
    cdef N_Vector y0, yp0, residual, y, yp
    cdef N_Vector dae_vars_id
    cdef long int N #problem size, i.e. len(y0) = N
    
    cdef int order, nsteps
    cdef double maxstep, first_step
    cdef bint constraints, exclude_algvar_from_error, out
    cdef int compute_initcond
    cdef double compute_initcond_t0
    cdef int mupper, mlower
    # ??? lband, uband, tcrit
    # ??? constraint_type, algebraic_var
    cdef void* _ida_mem
    cdef dict options
    cdef bint parallel_implementation
    cdef realtype t, t0
    cdef np.ndarray yy_tmp, yp_tmp, residual_tmp
        
    #cdef realtype *y0, *yprime0
    cdef JacFunction jac
    cdef ResFunction res
    
    # Functions
    cdef realtype _jacFn(self, Neq, tt, yy, yp, resvec, cj, jdata, JJ, 
                         tempv1, tempv2, tempv3)
    #def set_options(self, dict options)
    cpdef init_step(self, DTYPE_t t0, 
                 np.ndarray[DTYPE_t, ndim=1] y0, 
                 np.ndarray[DTYPE_t, ndim=1] ydot0)
    #def run_solver(self, np.ndarray[float, ndim=1] tspan, np.ndarray[float, ndim=1] y0, 
    #               np.ndarray[float, ndim=1] yp0)
    #def step(self, realtype t)
    
    