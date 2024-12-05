cimport numpy as np
from .c_sundials cimport N_Vector, SUNDlsMat, SUNMatrix, SUNContext

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


cdef class Shared_ErrHandler:
    cpdef evaluate(
        self,
        int line,
        bytes func,
        bytes file,
        bytes msg,
        int err_code,
        object user_data = *
    )


cdef class Shared_WrapErrHandler(Shared_ErrHandler):
    cdef object _err_handler
    cdef int with_userdata
    cdef int new_err_handler
    cpdef set_err_handler(self, object err_handler)


cdef class Shared_data:
    cdef bint parallel_implementation
    cdef object user_data
    cdef Shared_ErrHandler err_handler
    cdef object err_user_data


cdef class BaseSundialsSolver:
    cdef SUNContext sunctx
    cdef bint initialized
    cdef bint _old_api
    cdef bint _step_compute
    cdef bint _validate_flags
    cdef int verbosity
    cdef N_Vector atol
    cdef dict options
    cdef bint parallel_implementation
    cdef INDEX_TYPE_t N #problem size, i.e. len(y0) = N

    # Functions
    cpdef _create_suncontext(self)
