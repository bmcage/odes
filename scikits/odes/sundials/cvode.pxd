cimport numpy as np

cdef class CVODE:
    cpdef _init_step(self, DTYPE_t t0, 
                    np.ndarray[DTYPE_t, ndim=1] y0)

    cpdef _solve(self, np.ndarray[DTYPE_t, ndim=1] tspan, 
                    np.ndarray[DTYPE_t, ndim=1] y0, hook_fn = None):
