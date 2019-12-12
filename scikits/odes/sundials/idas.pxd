cimport numpy as np
from .c_sundials cimport N_Vector, realtype
from .common_defs cimport DTYPE_t

from .ida cimport (IDA_RhsFunction, IDA_WrapRhsFunction, IDA_RootFunction,
                   IDA_WrapRootFunction, IDA_JacRhsFunction,
                   IDA_WrapJacRhsFunction, IDA_PrecSetupFunction,
                   IDA_WrapPrecSetupFunction, IDA_PrecSolveFunction,
                   IDA_WrapPrecSolveFunction, IDA_JacTimesVecFunction,
                   IDA_WrapJacTimesVecFunction, IDA_JacTimesSetupFunction,
                   IDA_WrapJacTimesSetupFunction, IDA_ContinuationFunction,
                   IDA_ErrHandler, IDA_WrapErrHandler, IDA_data,
                   IDA)


cdef class IDAS_data(IDA_data):
    cdef np.ndarray yS_tmp, ySdot_tmp

cdef class IDAS(IDA):
    cdef N_Vector aStol
    cdef IDAS_data aux_dataS
    
    cdef int Ns   #sensitivity parameter size
