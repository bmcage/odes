cimport numpy as np
from .c_sundials cimport N_Vector, DlsMat

ctypedef np.float_t DTYPE_t

cdef inline int nv_s2ndarray(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a)
cdef inline int ndarray2nv_s(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a)

cdef inline int DlsMatd2ndarray(DlsMat m, np.ndarray a)
cdef inline int ndarray2DlsMatd(DlsMat m, np.ndarray a)

cdef ensure_numpy_float_array(object value)
