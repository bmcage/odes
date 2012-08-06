cdef extern from "sundials/sundials_types.h":
    ctypedef float realtype
    ctypedef unsigned int booleantype

cdef extern from "sundials/sundials_nvector.h":
    struct _generic_N_Vector_Ops:
        pass
    struct _generic_N_Vector:
        pass
    ctypedef _generic_N_Vector *N_Vector
    ctypedef _generic_N_Vector_Ops *N_Vector_Ops

    struct _generic_N_Vector_Ops:
        N_Vector    (*nvclone)(N_Vector)
        N_Vector    (*nvcloneempty)(N_Vector)
        void        (*nvdestroy)(N_Vector)
        void        (*nvspace)(N_Vector, long int *, long int *)
        realtype*   (*nvgetarraypointer)(N_Vector)
        void        (*nvsetarraypointer)(realtype *, N_Vector)
        void        (*nvlinearsum)(realtype, N_Vector, realtype, N_Vector, N_Vector)
        void        (*nvconst)(realtype, N_Vector)
        void        (*nvprod)(N_Vector, N_Vector, N_Vector)
        void        (*nvdiv)(N_Vector, N_Vector, N_Vector)
        void        (*nvscale)(realtype, N_Vector, N_Vector)
        void        (*nvabs)(N_Vector, N_Vector)
        void        (*nvinv)(N_Vector, N_Vector)
        void        (*nvaddconst)(N_Vector, realtype, N_Vector)
        realtype    (*nvdotprod)(N_Vector, N_Vector)
        realtype    (*nvmaxnorm)(N_Vector)
        realtype    (*nvwrmsnorm)(N_Vector, N_Vector)
        realtype    (*nvwrmsnormmask)(N_Vector, N_Vector, N_Vector)
        realtype    (*nvmin)(N_Vector)
        realtype    (*nvwl2norm)(N_Vector, N_Vector)
        realtype    (*nvl1norm)(N_Vector)
        void        (*nvcompare)(realtype, N_Vector, N_Vector)
        booleantype (*nvinvtest)(N_Vector, N_Vector)
        booleantype (*nvconstrmask)(N_Vector, N_Vector, N_Vector)
        realtype    (*nvminquotient)(N_Vector, N_Vector)

    struct _generic_N_Vector:
        void *content
        _generic_N_Vector_Ops *ops

    # * FUNCTIONS *
    N_Vector N_VClone(N_Vector w)
    N_Vector N_VCloneEmpty(N_Vector w)
    void N_VDestroy(N_Vector v)
    void N_VSpace(N_Vector v, long int *lrw, long int *liw)
    realtype *N_VGetArrayPointer(N_Vector v)
    void N_VSetArrayPointer(realtype *v_data, N_Vector v)
    void N_VLinearSum(realtype a, N_Vector x, realtype b, N_Vector y, N_Vector z)
    void N_VConst(realtype c, N_Vector z)
    void N_VProd(N_Vector x, N_Vector y, N_Vector z)
    void N_VDiv(N_Vector x, N_Vector y, N_Vector z)
    void N_VScale(realtype c, N_Vector x, N_Vector z)
    void N_VAbs(N_Vector x, N_Vector z)
    void N_VInv(N_Vector x, N_Vector z)
    void N_VAddConst(N_Vector x, realtype b, N_Vector z)
    realtype N_VDotProd(N_Vector x, N_Vector y)
    realtype N_VMaxNorm(N_Vector x)
    realtype N_VWrmsNorm(N_Vector x, N_Vector w)
    realtype N_VWrmsNormMask(N_Vector x, N_Vector w, N_Vector id)
    realtype N_VMin(N_Vector x)
    realtype N_VWL2Norm(N_Vector x, N_Vector w)
    realtype N_VL1Norm(N_Vector x)
    void N_VCompare(realtype c, N_Vector x, N_Vector z)
    booleantype N_VInvTest(N_Vector x, N_Vector z)
    booleantype N_VConstrMask(N_Vector c, N_Vector x, N_Vector m)
    realtype N_VMinQuotient(N_Vector num, N_Vector denom)

    N_Vector *N_VCloneEmptyVectorArray(int count, N_Vector w)
    N_Vector *N_VCloneVectorArray(int count, N_Vector w)
    void N_VDestroyVectorArray(N_Vector *vs, int count)

cdef extern from "nvector/nvector_serial.h":
    cdef struct _N_VectorContent_Serial:
        long int length
        booleantype own_data
        realtype *data
    ctypedef _N_VectorContent_Serial *N_VectorContent_Serial

    N_Vector N_VNew_Serial(long int vec_length)
    N_Vector N_VNewEmpty_Serial(long int vec_length)
    N_Vector N_VMake_Serial(long int vec_length, realtype *v_data)
    N_Vector *N_VCloneVectorArray_Serial(int count, N_Vector w)
    N_Vector *N_VCloneVectorArrayEmpty_Serial(int count, N_Vector w)
    void N_VDestroyVectorArray_Serial(N_Vector *vs, int count)

    void N_VPrint_Serial(N_Vector v)
    N_Vector N_VCloneEmpty_Serial(N_Vector w)
    N_Vector N_VClone_Serial(N_Vector w)
    void N_VDestroy_Serial(N_Vector v)
    void N_VSpace_Serial(N_Vector v, long int *lrw, long int *liw)
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

cdef extern from "sundials/sundials_lapack.h":
    pass
    # void dcopy_(int *n, double *x, int *inc_x, double *y, int *inc_y)
    # void dscal_(int *n, double *alpha, double *x, int *inc_x)

    # # Level-2 BLAS
    # void dgemv_(char *trans, int *m, int *n, double *alpha, double *a,
    #             int *lda, double *x, int *inc_x, double *beta, double *y,
    #             int *inc_y, int len_trans)
    # void dtrsv_(char *uplo, char *trans, char *diag, int *n,
    #             double *a, int *lda, double *x, int *inc_x,
    #             int len_uplo, int len_trans, int len_diag)

    # # Level-3 BLAS
    # void dsyrk_(char *uplo, char *trans, int *n, int *k,
    #             double *alpha, double *a, int *lda, double *beta,
    #             double *c, int *ldc, int len_uplo, int len_trans)

    # # LAPACK
    # void dgbtrf_(int *m, int *n, int *kl, int *ku,
    #              double *ab, int *ldab, int *ipiv, int *info)
    # void dgbtrs_(char *trans, int *n, int *kl, int *ku, int *nrhs,
    #              double *ab, int *ldab, int *ipiv, double *b, int *ldb,
    #              int *info, int len_trans)
    # void dgeqp3_(int *m, int *n, double *a, int *lda, int *jpvt, double *tau,
    #              double *work, int *lwork, int *info)
    # void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau, double *work,
    #              int *lwork, int *info)
    # void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info)
    # void dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda,
    #              int *ipiv, double *b, int *ldb, int *info, int len_trans)

    # void dormqr_(char *side, char *trans, int *m, int *n, int *k, double *a,
    #              int *lda, double *tau, double *c, int *ldc, double *work,
    #              int *lwork, int *info, int len_side, int len_trans)
    # void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info, int len_uplo)
    # void dpotrs_(char *uplo, int *n, int *nrhs, double *a, int *lda,
    #              double *b, int *ldb, int * info, int len_uplo)


cdef extern from "sundials/sundials_direct.h":
    enum: SUNDIALS_DENSE
    enum: SUNDIALS_BAND

    cdef struct _DlsMat:
        int type
        long int M
        long int N
        long int ldim
        long int mu
        long int ml
        long int s_mu
        realtype *data
        long int ldata
        realtype **cols

    ctypedef _DlsMat *DlsMat

    DlsMat NewDenseMat(long int M, long int N)
    DlsMat NewBandMat(long int N, long int mu, long int ml, long int smu)
    void DestroyMat(DlsMat A)
    int *NewIntArray(int N)
    long int *NewLintArray(long int N)
    realtype *NewRealArray(long int N)
    void DestroyArray(void *p)
    void AddIdentity(DlsMat A)
    void SetToZero(DlsMat A)
    void PrintMat(DlsMat A)

    # * Exported function prototypes (functions working on realtype**)
    realtype **newDenseMat(long int m, long int n)
    realtype **newBandMat(long int n, long int smu, long int ml)
    void destroyMat(realtype **a)
    int *newIntArray(int n)
    long int *newLintArray(long int n)
    realtype *newRealArray(long int m)
    void destroyArray(void *v)

# functions for accessing DlsMat data fields
cdef extern from "sundials_auxiliary/sundials_auxiliary.c":
    ctypedef realtype *nv_content_data_s

    cdef inline N_VectorContent_Serial nv_content_s(N_Vector v)
    cdef inline long int nv_length_s(N_VectorContent_Serial vc_s)
    cdef inline booleantype nv_own_data_s(N_VectorContent_Serial vc_s)
    cdef inline realtype* nv_data_s(N_VectorContent_Serial vc_s)
    cdef inline realtype get_nv_ith_s(nv_content_data_s vcd_s, int i)
    cdef inline void set_nv_ith_s(nv_content_data_s vcd_s, int i, realtype new_value)

    ctypedef realtype *DlsMat_col
    cdef inline int get_dense_N(DlsMat A)
    cdef inline int get_dense_M(DlsMat A)
    cdef inline int get_band_mu(DlsMat A)
    cdef inline int get_band_ml(DlsMat A)
    cdef inline realtype* get_dense_col(DlsMat A, int j)
    cdef inline void set_dense_col(DlsMat A, int j, realtype *data)
    cdef inline realtype get_dense_element(DlsMat A, int i, int j)
    cdef inline void set_dense_element(DlsMat A, int i, int j, realtype aij)
    cdef inline DlsMat_col get_band_col(DlsMat A, int j)
    cdef inline void set_band_col(DlsMat A, int j, realtype *data)
    cdef inline realtype get_band_col_elem(DlsMat_col col_j, int i, int j)
    cdef inline void set_band_col_elem(DlsMat_col col_j, int i, int j, realtype aij)
    cdef inline realtype get_band_element(DlsMat A, int i, int j)
    cdef inline void set_band_element(DlsMat A, int i, int j, realtype aij)

cdef extern from "sundials/sundials_band.h":
    long int BandGBTRF(DlsMat A, long int *p)
    long int bandGBTRF(realtype **a, long int n, long int mu, long int ml,
                       long int smu, long int *p)
    void BandGBTRS(DlsMat A, long int *p, realtype *b)
    void bandGBTRS(realtype **a, long int n, long int smu, long int ml,
                   long int *p, realtype *b)
    void BandCopy(DlsMat A, DlsMat B, long int copymu, long int copyml)
    void bandCopy(realtype **a, realtype **b, long int n, long int a_smu,
                  long int b_smu, long int copymu, long int copyml)
    void BandScale(realtype c, DlsMat A)
    void bandScale(realtype c, realtype **a, long int n, long int mu,
                   long int ml, long int smu)
    void bandAddIdentity(realtype **a, long int n, long int smu)

cdef extern from "sundials/sundials_dense.h":
    long int DenseGETRF(DlsMat A, long int *p)
    void DenseGETRS(DlsMat A, long int *p, realtype *b)

    long int denseGETRF(realtype **a, long int m, long int n, long int *p)
    void denseGETRS(realtype **a, long int n, long int *p, realtype *b)

    long int DensePOTRF(DlsMat A)
    void DensePOTRS(DlsMat A, realtype *b)

    long int densePOTRF(realtype **a, long int m)
    void densePOTRS(realtype **a, long int m, realtype *b)

    int DenseGEQRF(DlsMat A, realtype *beta, realtype *wrk)
    int DenseORMQR(DlsMat A, realtype *beta, realtype *vn, realtype *vm,
                   realtype *wrk)

    int denseGEQRF(realtype **a, long int m, long int n, realtype *beta, realtype *v)
    int denseORMQR(realtype **a, long int m, long int n, realtype *beta,
                   realtype *v, realtype *w, realtype *wrk)
    void DenseCopy(DlsMat A, DlsMat B)
    void denseCopy(realtype **a, realtype **b, long int m, long int n)
    void DenseScale(realtype c, DlsMat A)
    void denseScale(realtype c, realtype **a, long int m, long int n)
    void denseAddIdentity(realtype **a, long int n)

cdef extern from "sundials/sundials_iterative.h":
    enum:
        PREC_NONE
        PREC_LEFT
        PREC_RIGHT
        PREC_BOTH
    enum:
        MODIFIED_GS = 1
        CLASSICAL_GS = 2
    ctypedef int (*ATimesFn)(void *A_data, N_Vector v, N_Vector z)
    ctypedef int (*PSolveFn)(void *P_data, N_Vector r, N_Vector z, int lr)

    int ModifiedGS(N_Vector *v, realtype **h, int k, int p,
                   realtype *new_vk_norm)
    int ClassicalGS(N_Vector *v, realtype **h, int k, int p,
                    realtype *new_vk_norm, N_Vector temp, realtype *s)
    int QRfact(int n, realtype **h, realtype *q, int job)
    int QRsol(int n, realtype **h, realtype *q, realtype *b)

cdef extern from "sundials/sundials_spgmr.h":
    cdef struct _SpgmrMemRec:
        int l_max

        N_Vector *V
        realtype **Hes
        realtype *givens
        N_Vector xcor
        realtype *yg
        N_Vector vtemp
    ctypedef _SpgmrMemRec SpgmrMemRec
    ctypedef _SpgmrMemRec *SpgmrMem

    SpgmrMem SpgmrMalloc(int l_max, N_Vector vec_tmpl)
    int SpgmrSolve(SpgmrMem mem, void *A_data, N_Vector x, N_Vector b,
                   int pretype, int gstype, realtype delta,
                   int max_restarts, void *P_data, N_Vector s1,
                   N_Vector s2, ATimesFn atimes, PSolveFn psolve,
                   realtype *res_norm, int *nli, int *nps)

    enum: SPGMR_SUCCESS            #0  /* Converged                     */
    enum: SPGMR_RES_REDUCED        #1  /* Did not converge, but reduced
                                   #   /* norm of residual              */
    enum: SPGMR_CONV_FAIL          #2  /* Failed to converge            */
    enum: SPGMR_QRFACT_FAIL        #3  /* QRfact found singular matrix  */
    enum: SPGMR_PSOLVE_FAIL_REC    #4  /* psolve failed recoverably     */
    enum: SPGMR_ATIMES_FAIL_REC    #5  /* atimes failed recoverably     */
    enum: SPGMR_PSET_FAIL_REC      #6  /* pset faild recoverably        */

    enum: SPGMR_MEM_NULL          #-1  /* mem argument is NULL          */
    enum: SPGMR_ATIMES_FAIL_UNREC #-2  /* atimes returned failure flag  */
    enum: SPGMR_PSOLVE_FAIL_UNREC #-3  /* psolve failed unrecoverably   */
    enum: SPGMR_GS_FAIL           #-4  /* Gram-Schmidt routine faiuled  */
    enum: SPGMR_QRSOL_FAIL        #-5  /* QRsol found singular R        */
    enum: SPGMR_PSET_FAIL_UNREC   #-6  /* pset failed unrecoverably     */

    void SpgmrFree(SpgmrMem mem)

cdef extern from "sundials/sundials_spbcgs.h":
    ctypedef struct SpbcgMemRec:
        int l_max
        N_Vector r_star
        N_Vector r
        N_Vector p
        N_Vector q
        N_Vector u
        N_Vector Ap
        N_Vector vtemp
    ctypedef SpbcgMemRec *SpbcgMem

    SpbcgMem SpbcgMalloc(int l_max, N_Vector vec_tmpl)
    int SpbcgSolve(SpbcgMem mem, void *A_data, N_Vector x, N_Vector b,
                   int pretype, realtype delta, void *P_data, N_Vector sx,
                   N_Vector sb, ATimesFn atimes, PSolveFn psolve,
                   realtype *res_norm, int *nli, int *nps)

    enum: SPBCG_SUCCESS            #0  /* SPBCG algorithm converged          */
    enum: SPBCG_RES_REDUCED        #1  /* SPBCG did NOT converge, but the
                                   #      residual was reduced               */
    enum: SPBCG_CONV_FAIL          #2  /* SPBCG algorithm failed to converge */
    enum: SPBCG_PSOLVE_FAIL_REC    #3  /* psolve failed recoverably          */
    enum: SPBCG_ATIMES_FAIL_REC    #4  /* atimes failed recoverably          */
    enum: SPBCG_PSET_FAIL_REC      #5  /* pset faild recoverably             */
    enum: SPBCG_MEM_NULL          #-1  /* mem argument is NULL               */
    enum: SPBCG_ATIMES_FAIL_UNREC #-2  /* atimes returned failure flag       */
    enum: SPBCG_PSOLVE_FAIL_UNREC #-3  /* psolve failed unrecoverably        */
    enum: SPBCG_PSET_FAIL_UNREC   #-4  /* pset failed unrecoverably          */

    void SpbcgFree(SpbcgMem mem)

cdef extern from "sundials/sundials_sptfqmr.h":
    cdef struct SptfqmrMemRec:
        int l_max

        N_Vector r_star
        N_Vector q
        N_Vector d
        N_Vector v
        N_Vector p
        N_Vector *r
        N_Vector u
        N_Vector vtemp1
        N_Vector vtemp2
        N_Vector vtemp3
    ctypedef SptfqmrMemRec *SptfqmrMem

    SptfqmrMem SptfqmrMalloc(int l_max, N_Vector vec_tmpl)
    int SptfqmrSolve(SptfqmrMem mem, void *A_data, N_Vector x, N_Vector b,
                     int pretype, realtype delta, void *P_data, N_Vector sx,
                     N_Vector sb, ATimesFn atimes, PSolveFn psolve,
                     realtype *res_norm, int *nli, int *nps)

    enum: SPTFQMR_SUCCESS            #/* SPTFQMR algorithm converged          */
    enum: SPTFQMR_RES_REDUCED        #/* SPTFQMR did NOT converge, but the    */
    enum: SPTFQMR_CONV_FAIL          #/* SPTFQMR algorithm failed to converge */
    enum: SPTFQMR_PSOLVE_FAIL_REC    #/* psolve failed recoverably            */
    enum: SPTFQMR_ATIMES_FAIL_REC    #/* atimes failed recoverably            */
    enum: SPTFQMR_PSET_FAIL_REC      #/* pset faild recoverably               */

    enum: SPTFQMR_MEM_NULL           #/* mem argument is NULL                 */
    enum: SPTFQMR_ATIMES_FAIL_UNREC  #/* atimes returned failure flag         */
    enum: SPTFQMR_PSOLVE_FAIL_UNREC  #/* psolve failed unrecoverably          */
    enum: SPTFQMR_PSET_FAIL_UNREC    #/* pset failed unrecoverably            */

    void SptfqmrFree(SptfqmrMem mem)
