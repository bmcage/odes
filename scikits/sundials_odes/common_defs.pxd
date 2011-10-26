cimport numpy as np
from c_sundials cimport N_Vector

cdef class ResFunction:
    cpdef int evaluate(self, float t, 
                       np.ndarray[float, ndim=1] y,
                       np.ndarray[float, ndim=1] ydot,
                       np.ndarray[float, ndim=1] result)
        
cdef class JacFunction:
    cpdef np.ndarray[float, ndim=2] evaluate(self, float t, 
                                             np.ndarray[float, ndim=1] y,
                                             np.ndarray[float, ndim=1] ydot,
                                             float cj,
                                             np.ndarray[float, ndim=2] J)


cdef inline int nv_s2ndarray(N_Vector v, np.ndarray[float, ndim=1] a)
      
cdef inline int ndarray2nv_s(N_Vector v, np.ndarray[float, ndim=1] a)
    