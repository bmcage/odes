cimport numpy as np
from c_sundials cimport N_Vector, DlsMat

ctypedef np.float_t DTYPE_t

cdef class ResFunction:
    cpdef int evaluate(self, DTYPE_t t, 
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       np.ndarray[DTYPE_t, ndim=1] result,
                       object userdata = *)

cdef class WrapResFunction(ResFunction):
    cdef object _resfn
    cpdef set_resfn(self, object resfn)

cdef class JacFunction:
    cpdef np.ndarray[DTYPE_t, ndim=2] evaluate(self, DTYPE_t t, 
                                             np.ndarray[DTYPE_t, ndim=1] y,
                                             np.ndarray[DTYPE_t, ndim=1] ydot,
                                             DTYPE_t cj,
                                             np.ndarray[DTYPE_t, ndim=2] J)


cdef inline int nv_s2ndarray(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a)
cdef inline int ndarray2nv_s(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a)

cdef inline int DlsMatd2ndarray(DlsMat m, np.ndarray a)
cdef inline int ndarray2DlsMatd(DlsMat m, np.ndarray a)

cdef ensure_numpy_float_array(object value)
