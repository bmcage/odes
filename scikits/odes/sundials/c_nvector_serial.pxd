from libc.stdio cimport FILE
from .c_sundials cimport *

cdef extern from "nvector/nvector_serial.h":
    cdef struct _N_VectorContent_Serial:
        sunindextype length
        booleantype own_data
        realtype *data

    ctypedef _N_VectorContent_Serial *N_VectorContent_Serial

    N_Vector N_VNew_Serial(sunindextype vec_length)
    N_Vector N_VNewEmpty_Serial(sunindextype vec_length)
    N_Vector N_VMake_Serial(sunindextype vec_length, realtype *v_data)
    N_Vector *N_VCloneVectorArray_Serial(int count, N_Vector w)
    N_Vector *N_VCloneVectorArrayEmpty_Serial(int count, N_Vector w)
    void N_VDestroyVectorArray_Serial(N_Vector *vs, int count)
    sunindextype N_VGetLength_Serial(N_Vector v)
    void N_VPrint_Serial(N_Vector v)
    void N_VPrintFile_Serial(N_Vector v, FILE *outfile)

    N_Vector_ID N_VGetVectorID_Serial(N_Vector v)
    N_Vector N_VCloneEmpty_Serial(N_Vector w)
    N_Vector N_VClone_Serial(N_Vector w)
    void N_VDestroy_Serial(N_Vector v)
    void N_VSpace_Serial(N_Vector v, sunindextype *lrw, sunindextype *liw)
    realtype *N_VGetArrayPointer_Serial(N_Vector v)
    void N_VSetArrayPointer_Serial(realtype *v_data, N_Vector v)
    void N_VLinearSum_Serial(realtype a, N_Vector x, realtype b, N_Vector y, N_Vector z)
    void N_VConst_Serial(realtype c, N_Vector z)
    void N_VProd_Serial(N_Vector x, N_Vector y, N_Vector z)
    void N_VDiv_Serial(N_Vector x, N_Vector y, N_Vector z)
    void N_VScale_Serial(realtype c, N_Vector x, N_Vector z)
    void N_VAbs_Serial(N_Vector x, N_Vector z)
    void N_VInv_Serial(N_Vector x, N_Vector z)
    void N_VAddConst_Serial(N_Vector x, realtype b, N_Vector z)
    realtype N_VDotProd_Serial(N_Vector x, N_Vector y)
    realtype N_VMaxNorm_Serial(N_Vector x)
    realtype N_VWrmsNorm_Serial(N_Vector x, N_Vector w)
    realtype N_VWrmsNormMask_Serial(N_Vector x, N_Vector w, N_Vector id)
    realtype N_VMin_Serial(N_Vector x)
    realtype N_VWL2Norm_Serial(N_Vector x, N_Vector w)
    realtype N_VL1Norm_Serial(N_Vector x)
    void N_VCompare_Serial(realtype c, N_Vector x, N_Vector z)
    booleantype N_VInvTest_Serial(N_Vector x, N_Vector z)
    booleantype N_VConstrMask_Serial(N_Vector c, N_Vector x, N_Vector m)
    realtype N_VMinQuotient_Serial(N_Vector num, N_Vector denom)
    # Macros
    sunindextype NV_LENGTH_S(N_Vector vc_s)
    realtype* NV_DATA_S(N_Vector vc_s)

