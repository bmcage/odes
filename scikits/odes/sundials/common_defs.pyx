import numpy as np
cimport numpy as np
import inspect
from c_sundials cimport (N_Vector, nv_content_data_s, nv_content_s, nv_length_s,
                        nv_data_s, get_nv_ith_s, set_nv_ith_s, get_dense_col,
                        get_dense_N, set_dense_element,
                        DlsMat)

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
        self.with_userdata = 0
        nrarg = len(inspect.getargspec(resfn)[0])
        if nrarg > 5:
            # hopefully a class method
            self.with_userdata = 1
        elif nrarg == 5 and inspect.isfunction(resfn):
            self.with_userdata = 1
        self._resfn = resfn

    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       np.ndarray[DTYPE_t, ndim=1] result,
                       object userdata = None):
        if self.with_userdata == 1:
            self._resfn(t, y, ydot, result, userdata)
        else:
            self._resfn(t, y, ydot, result)
        return 0

cdef class RhsFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       object userdata = None):
        return 0

cdef class WrapRhsFunction(RhsFunction):
    cpdef set_rhsfn(self, object rhsfn):
        """
        set some rhs equations as a RhsFunction executable class
        """
        self.with_userdata = 0
        nrarg = len(inspect.getargspec(rhsfn)[0])
        if nrarg > 4:
            #hopefully a class method, self gives 5 arg!
            self.with_userdata = 1
        elif nrarg == 4 and inspect.isfunction(rhsfn):
            self.with_userdata = 1
        self._rhsfn = rhsfn

    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       object userdata = None):
        if self.with_userdata == 1:
            self._rhsfn(t, y, ydot, userdata)
        else:
            self._rhsfn(t, y, ydot)
        return 0

cdef class JacFunction:
    cpdef int evaluate(self, DTYPE_t t, 
                                             np.ndarray[DTYPE_t, ndim=1] y,
                                             np.ndarray[DTYPE_t, ndim=1] ydot,
                                             DTYPE_t cj,
                                             np.ndarray J):
        """
        Returns the Jacobi matrix (for dense the full matrix, for band only
        bands. Result has to be stored in the variable J, which is preallocated
        to the corresponding size.
            
        This is a generic class, you should subclass is for the problem specific
        purposes."
        """
        return 0

cdef class WrapJacFunction(JacFunction):
    cpdef set_jacfn(self, object jacfn):
        """
        set some jacobian equations as a JacFunction executable class
        """
##        self.with_userdata = 0
##        nrarg = len(inspect.getargspec(jacfn)[0])
##        if nrarg > 6:
##            #hopefully a class method, self gives 5 arg!
##            self.with_userdata = 1
##        elif nrarg == 6 and inspect.isfunction(jacfn):
##            self.with_userdata = 1
        self._jacfn = jacfn

    cpdef int evaluate(self, DTYPE_t t, 
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       DTYPE_t cj,
                       np.ndarray J):
        """
        Returns the Jacobi matrix (for dense the full matrix, for band only
        bands. Result has to be stored in the variable J, which is preallocated
        to the corresponding size.
            
        This is a generic class, you should subclass is for the problem specific
        purposes."
        """
##        if self.with_userdata == 1:
##            self._jacfn(t, y, ydot, cj, J, userdata)
##        else:
##            self._jacfn(t, y, ydot, cj, J)
        self._jacfn(t, y, ydot, cj, J)
        return 0

cdef inline int nv_s2ndarray(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a):
    """ copy a serial N_Vector v to a nympy array a """
    cdef unsigned int N, i
    N = nv_length_s(nv_content_s(v))
    cdef nv_content_data_s v_data = nv_data_s(nv_content_s(v))
    
    for i in range(N):
      a[i] = get_nv_ith_s(v_data, i)
      
cdef inline int ndarray2nv_s(N_Vector v, np.ndarray[DTYPE_t, ndim=1] a):
    """ copy a numpy array a to a serial N_Vector v t"""
    cdef unsigned int N, i
    N = nv_length_s(nv_content_s(v))
    cdef nv_content_data_s v_data = nv_data_s(nv_content_s(v))
    
    for i in range(N):
      set_nv_ith_s(v_data, i, a[i])

cdef inline int DlsMatd2ndarray(DlsMat m, np.ndarray a):
    """ copy a Dense DlsMat m to a nympy array a """
    cdef unsigned int N, i, j
    cdef nv_content_data_s v_col
    
    N = get_dense_N(m)
    
    for i in range(N):
        v_col = get_dense_col(m, i)
        for j in range(N):
            a[i,j] = get_nv_ith_s(v_col, j)

cdef inline int ndarray2DlsMatd(DlsMat m, np.ndarray a):
    """ copy a nympy array a to a Dense DlsMat m"""
    cdef unsigned int N, i, j
    cdef nv_content_data_s v_col
    
    N = get_dense_N(m)
    
    for i in range(N):
        for j in range(N):
            set_dense_element(m, i, j, a[i,j])

cdef ensure_numpy_float_array(object value):
    try:
        if (type(value) == float or type(value) == int
            or type(value) == np.float
            or type(value) == np.float32
            or type(value) == np.float64
            or type(value) == np.float128
            or type(value) == np.int
            or type(value) == np.int8
            or type(value) == np.int16
            or type(value) == np.int32
            or type(value) == np.int64):

            return np.array([value, ], float)
        else:
            return np.asarray(value, float)
    except:
        raise ValueError('ensure_numpy_float_array: value not a number or sequence of numbers: %s' % value)

