cimport numpy as np
from c_sundials cimport N_Vector, DlsMat

ctypedef np.float_t DTYPE_t

cdef class IDA_RhsFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       np.ndarray[DTYPE_t, ndim=1] result,
                       object userdata = *)

cdef class IDA_WrapRhsFunction(IDA_RhsFunction):
    cdef object _resfn
    cdef int with_userdata
    cpdef set_resfn(self, object resfn)

cdef class CV_RhsFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       object userdata = *)
cdef class CV_WrapRhsFunction(CV_RhsFunction):
    cpdef object _rhsfn
    cdef int with_userdata
    cpdef set_rhsfn(self, object rhsfn)

cdef class IDA_RootFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       np.ndarray[DTYPE_t, ndim=1] g,
                       object userdata = *)
cdef class IDA_WrapRootFunction(IDA_RootFunction):
    cpdef object _rootfn
    cdef int with_userdata
    cpdef set_rootfn(self, object rootfn)

cdef class CV_RootFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] g,
                       object userdata = *)
cdef class CV_WrapRootFunction(CV_RootFunction):
    cpdef object _rootfn
    cdef int with_userdata
    cpdef set_rootfn(self, object rootfn)

cdef class IDA_JacRhsFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       DTYPE_t cj,
                       np.ndarray[DTYPE_t, ndim=2] J)

cdef class IDA_WrapJacRhsFunction(IDA_JacRhsFunction):
    cpdef object _jacfn
    cdef int with_userdata
    cpdef set_jacfn(self, object jacfn)

cdef class CV_JacRhsFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=2] J)

cdef class CV_WrapJacRhsFunction(CV_JacRhsFunction):
    cpdef object _jacfn
    cdef int with_userdata
    cpdef set_jacfn(self, object jacfn)

cdef inline int nv_s2ndarray(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a)
cdef inline int ndarray2nv_s(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a)

cdef inline int DlsMatd2ndarray(DlsMat m, np.ndarray a)
cdef inline int ndarray2DlsMatd(DlsMat m, np.ndarray a)

cdef ensure_numpy_float_array(object value)
