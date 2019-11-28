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
    
    
    int N_VLinearCombination_Serial(int nvec, realtype* c, N_Vector* V,
                                    N_Vector z)
    int N_VScaleAddMulti_Serial(int nvec, realtype* a, N_Vector x,
                                N_Vector* Y, N_Vector* Z)
    int N_VDotProdMulti_Serial(int nvec, N_Vector x, N_Vector* Y, 
                               realtype* dotprods)

    int N_VLinearSumVectorArray_Serial(int nvec, realtype a, N_Vector* X,
                                       realtype b, N_Vector* Y, N_Vector* Z)
    int N_VScaleVectorArray_Serial(int nvec, realtype* c,
                                   N_Vector* X, N_Vector* Z)
    int N_VConstVectorArray_Serial(int nvecs, realtype c, N_Vector* Z)
    int N_VWrmsNormVectorArray_Serial(int nvecs, N_Vector* X,
                                      N_Vector* W, realtype* nrm)
    int N_VWrmsNormMaskVectorArray_Serial(int nvecs, N_Vector* X,
                                          N_Vector* W, N_Vector id,
                                          realtype* nrm)
    int N_VScaleAddMultiVectorArray_Serial(int nvec, int nsum, realtype* a,
                                           N_Vector* X, N_Vector** Y,
                                           N_Vector** Z)
    int N_VLinearCombinationVectorArray_Serial(int nvec, int nsum,
                                               realtype* c,
                                               N_Vector** X, N_Vector* Z)

    realtype N_VWSqrSumLocal_Serial(N_Vector x, N_Vector w)
    realtype N_VWSqrSumMaskLocal_Serial(N_Vector x, N_Vector w, N_Vector id)
  
    int N_VEnableFusedOps_Serial(N_Vector v, booleantype tf)

    int N_VEnableLinearCombination_Serial(N_Vector v, booleantype tf)
    int N_VEnableScaleAddMulti_Serial(N_Vector v, booleantype tf)
    int N_VEnableDotProdMulti_Serial(N_Vector v, booleantype tf)

    int N_VEnableLinearSumVectorArray_Serial(N_Vector v, booleantype tf)
    int N_VEnableScaleVectorArray_Serial(N_Vector v, booleantype tf)
    int N_VEnableConstVectorArray_Serial(N_Vector v, booleantype tf)
    int N_VEnableWrmsNormVectorArray_Serial(N_Vector v, booleantype tf)
    int N_VEnableWrmsNormMaskVectorArray_Serial(N_Vector v, booleantype tf)
    int N_VEnableScaleAddMultiVectorArray_Serial(N_Vector v, booleantype tf)
    int N_VEnableLinearCombinationVectorArray_Serial(N_Vector v, booleantype tf)

    # Macros
    sunindextype NV_LENGTH_S(N_Vector vc_s)
    realtype* NV_DATA_S(N_Vector vc_s)

