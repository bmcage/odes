cimport numpy as np
from .c_sundials cimport N_Vector, DlsMat, SUNMatrix

include "sundials_config.pxi"

IF SUNDIALS_FLOAT_TYPE == "single":
    ctypedef np.float_t DTYPE_t
ELIF SUNDIALS_FLOAT_TYPE == "double":
    ctypedef np.double_t DTYPE_t
ELIF SUNDIALS_FLOAT_TYPE == "extended":
    ctypedef np.longdouble_t DTYPE_t
ELSE:
    # fall back to double
    ctypedef np.double_t DTYPE_t

IF SUNDIALS_INDEX_SIZE == "32":
    ctypedef np.int32_t INDEX_TYPE_t
ELIF SUNDIALS_INDEX_SIZE == "64":
    ctypedef np.int64_t INDEX_TYPE_t
ELSE:
    ctypedef np.int64_t INDEX_TYPE_t

cdef int nv_s2ndarray(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a) except? -1
cdef int ndarray2nv_s(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a) except? -1

cdef int SUNMatrix2ndarray(SUNMatrix m, np.ndarray a) except? -1
cdef int ndarray2SUNMatrix(SUNMatrix m, np.ndarray a) except? -1

cdef ensure_numpy_float_array(object value)
