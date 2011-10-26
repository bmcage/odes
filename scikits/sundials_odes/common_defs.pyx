#import numpy as np
cimport numpy as np
from c_sundials cimport N_Vector, nv_content_data_s, nv_content_s, nv_length_s, nv_data_s, get_nv_ith_s, set_nv_ith_s

cdef class ResFunction:
    cpdef int evaluate(self, float t, 
                       np.ndarray[float, ndim=1] y,
                       np.ndarray[float, ndim=1] ydot,
                       #void *userdata = NULL
                       np.ndarray[float, ndim=1] result):
        return np.array([])
        
cdef class JacFunction:
    cpdef np.ndarray[float, ndim=2]  evaluate(self, float t, 
                                             np.ndarray[float, ndim=1] y,
                                             np.ndarray[float, ndim=1] ydot,
                                             float cj,
                                             np.ndarray[float, ndim=2] J):
        """
            Returns the Jacobi matrix (for dense the full matrix, for band only
            bands. Result has to be stored in the variable J, which is preallocated
            to the corresponding size.
        """
        raise NotImplemented("You should subclass this generic class for specific\
                              purposes of the problem.")
        return 0

cdef inline int nv_s2ndarray(N_Vector v, np.ndarray[float, ndim=1] a):
    cdef unsigned int N, i
    N = nv_length_s(nv_content_s(v))
    cdef nv_content_data_s v_data = nv_data_s(nv_content_s(v))
    
    for i in range(N):
      a[i] = get_nv_ith_s(v_data, i)
      
cdef inline int ndarray2nv_s(N_Vector v, np.ndarray[float, ndim=1] a):
    cdef unsigned int N, i
    N = nv_length_s(nv_content_s(v))
    cdef nv_content_data_s v_data = nv_data_s(nv_content_s(v))
    
    for i in range(N):
      set_nv_ith_s(v_data, i, a[i])
    