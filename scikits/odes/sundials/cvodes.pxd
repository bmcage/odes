#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

cimport numpy as np
from .c_sundials cimport N_Vector, realtype
from .common_defs cimport DTYPE_t, INDEX_TYPE_t

from .cvode cimport (CV_RhsFunction, CV_WrapRhsFunction, CV_RootFunction,
                       CV_WrapRootFunction, CV_JacRhsFunction,
                       CV_WrapJacRhsFunction, CV_PrecSetupFunction,
                       CV_WrapPrecSetupFunction,
                       CV_PrecSolveFunction, CV_WrapPrecSolveFunction,
                       CV_JacTimesVecFunction, CV_WrapJacTimesVecFunction,
                       CV_JacTimesSetupFunction, CV_WrapJacTimesSetupFunction,
                       CV_ContinuationFunction, CV_ErrHandler, 
                       CV_WrapErrHandler, CV_data, CVODE)

cdef class CVS_data(CV_data):
    cdef np.ndarray yS_tmp, ySdot_tmp

cdef class CVODES(CVODE):
    cdef N_Vector aStol
    cdef CVS_data aux_dataS

    cdef int Ns   #sensitivity parameter size
