#import numpy as np
cimport numpy as np
from c_sundials cimport (N_Vector, nv_content_data_s, nv_content_s, nv_length_s,
                        nv_data_s, get_nv_ith_s, set_nv_ith_s)

cdef class ResFunction:
    cpdef int evaluate(self, DTYPE_t t, 
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       np.ndarray[DTYPE_t, ndim=1] result,
                       object userdata = None):
        return 0

cdef class WrapResFunction(ResFunction):
    cpdef set_resfn(self, object resfn):
        """
        set some residual equations as a ResFunction executable class
        """
        self._resfn = resfn

    cpdef int evaluate(self, DTYPE_t t, 
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       np.ndarray[DTYPE_t, ndim=1] result,
                       object userdata = None):
        self._resfn(t, y, ydot, result)
        return 0
    
cdef class JacFunction:
    cpdef np.ndarray[DTYPE_t, ndim=2]  evaluate(self, DTYPE_t t, 
                                             np.ndarray[DTYPE_t, ndim=1] y,
                                             np.ndarray[DTYPE_t, ndim=1] ydot,
                                             DTYPE_t cj,
                                             np.ndarray[DTYPE_t, ndim=2] J):
        """
        Returns the Jacobi matrix (for dense the full matrix, for band only
        bands. Result has to be stored in the variable J, which is preallocated
        to the corresponding size.
            
        This is a generic class, you should subclass is for the problem specific
        purposes."
        """
        return 0

cdef inline int nv_s2ndarray(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a):
    cdef unsigned int N, i
    N = nv_length_s(nv_content_s(v))
    cdef nv_content_data_s v_data = nv_data_s(nv_content_s(v))
    
    for i in range(N):
      a[i] = get_nv_ith_s(v_data, i)
      
cdef inline int ndarray2nv_s(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a):
    cdef unsigned int N, i
    N = nv_length_s(nv_content_s(v))
    cdef nv_content_data_s v_data = nv_data_s(nv_content_s(v))
    
    for i in range(N):
      set_nv_ith_s(v_data, i, a[i])
    