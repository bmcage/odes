from libc.stdio cimport FILE

cdef extern from "sundials/sundials_types.h":
    ctypedef float realtype
    ctypedef unsigned int booleantype
    ctypedef long sunindextype

cdef extern from "sundials/sundials_nvector.h":
    cdef enum N_Vector_ID:
        SUNDIALS_NVEC_SERIAL,
        SUNDIALS_NVEC_PARALLEL,
        SUNDIALS_NVEC_OPENMP,
        SUNDIALS_NVEC_PTHREADS,
        SUNDIALS_NVEC_PARHYP,
        SUNDIALS_NVEC_PETSC,
        SUNDIALS_NVEC_CUDA,
        SUNDIALS_NVEC_RAJA,
        SUNDIALS_NVEC_CUSTOM

    struct _generic_N_Vector_Ops:
        pass
    struct _generic_N_Vector:
        pass
    ctypedef _generic_N_Vector *N_Vector
    ctypedef _generic_N_Vector_Ops *N_Vector_Ops

    struct _generic_N_Vector_Ops:
        N_Vector_ID (*nvgetvectorid)(N_Vector)
        N_Vector    (*nvclone)(N_Vector)
        N_Vector    (*nvcloneempty)(N_Vector)
        void        (*nvdestroy)(N_Vector)
        void        (*nvspace)(N_Vector, sunindextype *, sunindextype *)
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

    # * FUNCTIONS *
    N_Vector_ID N_VGetVectorID(N_Vector w)
    N_Vector N_VClone(N_Vector w)
    N_Vector N_VCloneEmpty(N_Vector w)
    void N_VDestroy(N_Vector v)
    void N_VSpace(N_Vector v, sunindextype *lrw, sunindextype *liw)
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

cdef extern from "sundials/sundials_matrix.h":
    cdef enum SUNMatrix_ID:
        SUNMATRIX_DENSE,
        SUNMATRIX_BAND,
        SUNMATRIX_SPARSE,
        SUNMATRIX_CUSTOM

    struct _generic_SUNMatrix_Ops:
        pass
    struct _generic_SUNMatrix:
        pass
    ctypedef _generic_SUNMatrix *SUNMatrix
    ctypedef _generic_SUNMatrix_Ops *SUNMatrix_Ops

    struct _generic_SUNMatrix_Ops:
        SUNMatrix_ID (*getid)(SUNMatrix)
        SUNMatrix    (*clone)(SUNMatrix)
        void         (*destroy)(SUNMatrix)
        int          (*zero)(SUNMatrix)
        int          (*copy)(SUNMatrix, SUNMatrix)
        int          (*scaleadd)(realtype, SUNMatrix, SUNMatrix)
        int          (*scaleaddi)(realtype, SUNMatrix)
        int          (*matvec)(SUNMatrix, N_Vector, N_Vector)
        int          (*space)(SUNMatrix, long int*, long int*)

    #struct _generic_SUNMatrix:
    #    void *content
    #    struct _generic_SUNMatrix_Ops *ops

    # * FUNCTIONS *
    SUNMatrix_ID SUNMatGetID(SUNMatrix A)
    SUNMatrix SUNMatClone(SUNMatrix A)
    void SUNMatDestroy(SUNMatrix A)
    int SUNMatZero(SUNMatrix A)
    int SUNMatCopy(SUNMatrix A, SUNMatrix B)
    int SUNMatScaleAdd(realtype c, SUNMatrix A, SUNMatrix B)
    int SUNMatScaleAddI(realtype c, SUNMatrix A)
    int SUNMatMatvec(SUNMatrix A, N_Vector x, N_Vector y)
    int SUNMatSpace(SUNMatrix A, long int *lenrw, long int *leniw)

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
    ctypedef int (*PSetupFn)(void *P_data)
    ctypedef int (*PSolveFn)(void *P_data, N_Vector r, N_Vector z,
                             realtype tol, int lr)

    int ModifiedGS(N_Vector *v, realtype **h, int k, int p,
                   realtype *new_vk_norm)
    int ClassicalGS(N_Vector *v, realtype **h, int k, int p,
                    realtype *new_vk_norm, N_Vector temp, realtype *s)
    int QRfact(int n, realtype **h, realtype *q, int job)
    int QRsol(int n, realtype **h, realtype *q, realtype *b)

# We don't support KLU for now
#cdef extern from "sundials/sundials_klu_impl.h":
#    cdef struct KLUDataRec:
#        klu_symbolic *s_Symbolic
#        klu_numeric  *s_Numeric
#        klu_common    s_Common
#        int           s_ordering
#    ctypedef KLUDataRec *KLUData

cdef extern from "sundials/sundials_linearsolver.h":

    cdef enum SUNLinearSolver_Type:
        SUNLINEARSOLVER_DIRECT,
        SUNLINEARSOLVER_ITERATIVE,
        SUNLINEARSOLVER_CUSTOM

    struct _generic_SUNLinearSolver_Ops:
        pass
    struct _generic_SUNLinearSolver:
        pass
    ctypedef _generic_SUNLinearSolver *SUNLinearSolver
    ctypedef _generic_SUNLinearSolver_Ops *SUNLinearSolver_Ops

    struct _generic_SUNLinearSolver_Ops:
        SUNLinearSolver_Type (*gettype)(SUNLinearSolver)
        int                  (*setatimes)(SUNLinearSolver, void*, ATimesFn)
        int                  (*setpreconditioner)(SUNLinearSolver, void*,
                                                PSetupFn, PSolveFn)
        int                  (*setscalingvectors)(SUNLinearSolver,
                                                N_Vector, N_Vector)
        int                  (*initialize)(SUNLinearSolver)
        int                  (*setup)(SUNLinearSolver, SUNMatrix)
        int                  (*solve)(SUNLinearSolver, SUNMatrix, N_Vector,
                                    N_Vector, realtype)
        int                  (*numiters)(SUNLinearSolver)
        realtype             (*resnorm)(SUNLinearSolver)
        long int             (*lastflag)(SUNLinearSolver)
        int                  (*space)(SUNLinearSolver, long int*, long int*)
        N_Vector             (*resid)(SUNLinearSolver)
        int                  (*free)(SUNLinearSolver)

    #struct _generic_SUNLinearSolver:
    #    void *content
    #    struct _generic_SUNLinearSolver_Ops *ops

    SUNLinearSolver_Type SUNLinSolGetType(SUNLinearSolver S)
    int SUNLinSolSetATimes(SUNLinearSolver S, void* A_data,
                           ATimesFn ATimes)
    int SUNLinSolSetPreconditioner(SUNLinearSolver S, void* P_data,
                                   PSetupFn Pset, PSolveFn Psol)
    int SUNLinSolSetScalingVectors(SUNLinearSolver S, N_Vector s1,
                                   N_Vector s2)
    int SUNLinSolInitialize(SUNLinearSolver S)
    int SUNLinSolSetup(SUNLinearSolver S, SUNMatrix A)
    int SUNLinSolSolve(SUNLinearSolver S, SUNMatrix A, N_Vector x,
                       N_Vector b, realtype tol)
    int SUNLinSolNumIters(SUNLinearSolver S)
    realtype SUNLinSolResNorm(SUNLinearSolver S)
    N_Vector SUNLinSolResid(SUNLinearSolver S)
    long int SUNLinSolLastFlag(SUNLinearSolver S)
    int SUNLinSolSpace(SUNLinearSolver S, long int *lenrwLS,
                                       long int *leniwLS)
    int SUNLinSolFree(SUNLinearSolver S)


cdef extern from "sundials/sundials_direct.h":
    enum: SUNDIALS_DENSE
    enum: SUNDIALS_BAND

    cdef struct _DlsMat:
        int type
        sunindextype M
        sunindextype N
        sunindextype ldim
        sunindextype mu
        sunindextype ml
        sunindextype s_mu
        realtype *data
        sunindextype ldata
        realtype **cols

    ctypedef _DlsMat *DlsMat

    DlsMat NewDenseMat(sunindextype M, sunindextype N)
    DlsMat NewBandMat(sunindextype N, sunindextype mu, sunindextype ml, sunindextype smu)
    void DestroyMat(DlsMat A)
    int *NewIntArray(int N)
    sunindextype *NewIndexArray(sunindextype N)
    realtype *NewRealArray(long int N)
    void DestroyArray(void *p)
    void AddIdentity(DlsMat A)
    void SetToZero(DlsMat A)
    void PrintMat(DlsMat A, FILE *outfile)

    # * Exported function prototypes (functions working on realtype**)
    realtype **newDenseMat(sunindextype m, sunindextype n)
    realtype **newBandMat(sunindextype n, sunindextype smu, sunindextype ml)
    void destroyMat(realtype **a)
    int *newIntArray(int n)
    sunindextype *newIndexArray(sunindextype n)
    realtype *newRealArray(sunindextype m)
    void destroyArray(void *v)

cdef extern from "sundials/sundials_band.h":
    sunindextype BandGBTRF(DlsMat A, sunindextype *p)
    sunindextype bandGBTRF(realtype **a, sunindextype n, sunindextype mu, sunindextype ml,
                       sunindextype smu, sunindextype *p)
    void BandGBTRS(DlsMat A, sunindextype *p, realtype *b)
    void bandGBTRS(realtype **a, sunindextype n, sunindextype smu, sunindextype ml,
                   sunindextype *p, realtype *b)
    void BandCopy(DlsMat A, DlsMat B, sunindextype copymu, sunindextype copyml)
    void bandCopy(realtype **a, realtype **b, sunindextype n, sunindextype a_smu,
                  sunindextype b_smu, sunindextype copymu, sunindextype copyml)
    void BandScale(realtype c, DlsMat A)
    void bandScale(realtype c, realtype **a, sunindextype n, sunindextype mu,
                   sunindextype ml, sunindextype smu)
    void bandAddIdentity(realtype **a, sunindextype n, sunindextype smu)
    void BandMatvec(DlsMat A, realtype *x, realtype *y)
    void bandMatvec(realtype **a, realtype *x, realtype *y, sunindextype n,
                    sunindextype mu, sunindextype ml, sunindextype smu)

cdef extern from "sundials/sundials_dense.h":
    sunindextype DenseGETRF(DlsMat A, sunindextype *p)
    void DenseGETRS(DlsMat A, sunindextype *p, realtype *b)

    sunindextype denseGETRF(realtype **a, sunindextype m, sunindextype n, sunindextype *p)
    void denseGETRS(realtype **a, sunindextype n, sunindextype *p, realtype *b)

    sunindextype DensePOTRF(DlsMat A)
    void DensePOTRS(DlsMat A, realtype *b)

    sunindextype densePOTRF(realtype **a, sunindextype m)
    void densePOTRS(realtype **a, sunindextype m, realtype *b)

    int DenseGEQRF(DlsMat A, realtype *beta, realtype *wrk)
    int DenseORMQR(DlsMat A, realtype *beta, realtype *vn, realtype *vm,
                   realtype *wrk)

    int denseGEQRF(realtype **a, sunindextype m, sunindextype n, realtype *beta, realtype *v)
    int denseORMQR(realtype **a, sunindextype m, sunindextype n, realtype *beta,
                   realtype *v, realtype *w, realtype *wrk)
    void DenseCopy(DlsMat A, DlsMat B)
    void denseCopy(realtype **a, realtype **b, sunindextype m, sunindextype n)
    void DenseScale(realtype c, DlsMat A)
    void denseScale(realtype c, realtype **a, sunindextype m, sunindextype n)
    void denseAddIdentity(realtype **a, sunindextype n)
    void DenseMatvec(DlsMat A, realtype *x, realtype *y)
    void denseMatvec(realtype **a, realtype *x, realtype *y, sunindextype m, sunindextype n)

cdef extern from "sundials/sundials_pcg.h":
    ctypedef struct _PcgMemRec:
        int l_max
        N_Vector r
        N_Vector p
        N_Vector z
        N_Vector Ap
    ctypedef _PcgMemRec PcgMemRec
    ctypedef _PcgMemRec *PcgMem

    PcgMem PcgMalloc(int l_max, N_Vector vec_tmpl)
    int PcgSolve(PcgMem mem, void *A_data, N_Vector x, N_Vector b, int pretype,
                 realtype delta, void *P_data, N_Vector w, ATimesFn atimes,
                 PSolveFn psolve, realtype *res_norm, int *nli, int *nps)
    void PcgFree(PcgMem mem)

    enum: PCG_SUCCESS           #  0  /* PCG algorithm converged          */
    enum: PCG_RES_REDUCED       #  1  /* PCG did NOT converge, but the
                                #        residual was reduced             */
    enum: PCG_CONV_FAIL         #  2  /* PCG algorithm failed to converge */
    enum: PCG_PSOLVE_FAIL_REC   #  3  /* psolve failed recoverably        */
    enum: PCG_ATIMES_FAIL_REC   #  4  /* atimes failed recoverably        */
    enum: PCG_PSET_FAIL_REC     #  5  /* pset faild recoverably           */

    enum: PCG_MEM_NULL          # -1  /* mem argument is NULL             */
    enum: PCG_ATIMES_FAIL_UNREC # -2  /* atimes returned failure flag     */
    enum: PCG_PSOLVE_FAIL_UNREC # -3  /* psolve failed unrecoverably      */
    enum: PCG_PSET_FAIL_UNREC   # -4  /* pset failed unrecoverably        */

cdef extern from "sundials/sundials_sparse.h":

    enum: CSC_MAT #0
    enum: CSR_MAT #1

    ctypedef struct _SlsMat:
       int M
       int N
       int NNZ
       int NP
       realtype *data
       int sparsetype
       int *indexvals
       int *indexptrs
       int **rowvals
       int **colptrs
       int **colvals
       int **rowptrs
    ctypedef _SlsMat *SlsMat

    SlsMat SparseNewMat(int M, int N, int NNZ, int sparsetype)
    SlsMat SparseFromDenseMat(const DlsMat A, int sparsetype)
    int SparseDestroyMat(SlsMat A)
    int SparseSetMatToZero(SlsMat A)
    int SparseCopyMat(const SlsMat A, SlsMat B)
    int SparseScaleMat(realtype b, SlsMat A)
    int SparseAddIdentityMat(SlsMat A)
    int SparseAddMat(SlsMat A, const SlsMat B)
    int SparseReallocMat(SlsMat A)
    int SparseMatvec(const SlsMat A, const realtype *x, realtype *y)
    void SparsePrintMat(const SlsMat A, FILE* outfile)

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

cdef extern from "sundials/sundials_spfgmr.h":
    ctypedef struct _SpfgmrMemRec:
        int l_max
        N_Vector *V
        N_Vector *Z
        realtype **Hes
        realtype *givens
        N_Vector xcor
        realtype *yg
        N_Vector vtemp
    ctypedef _SpfgmrMemRec SpfgmrMemRec
    ctypedef _SpfgmrMemRec *SpfgmrMem

    SpfgmrMem SpfgmrMalloc(int l_max, N_Vector vec_tmpl)
    int SpfgmrSolve(SpfgmrMem mem, void *A_data, N_Vector x, N_Vector b,
                    int pretype, int gstype, realtype delta, int max_restarts,
                    int maxit, void *P_data, N_Vector s1, N_Vector s2,
                    ATimesFn atimes, PSolveFn psolve, realtype *res_norm,
                    int *nli, int *nps)
    void SpfgmrFree(SpfgmrMem mem)

    enum: SPFGMR_SUCCESS           #  0  /* Converged                     */
    enum: SPFGMR_RES_REDUCED       #  1  /* Did not converge, but reduced
                                   #        norm of residual              */
    enum: SPFGMR_CONV_FAIL         # 2  /* Failed to converge            */
    enum: SPFGMR_QRFACT_FAIL       # 3  /* QRfact found singular matrix  */
    enum: SPFGMR_PSOLVE_FAIL_REC   # 4  /* psolve failed recoverably     */
    enum: SPFGMR_ATIMES_FAIL_REC   # 5  /* atimes failed recoverably     */
    enum: SPFGMR_PSET_FAIL_REC     # 6  /* pset faild recoverably        */

    enum: SPFGMR_MEM_NULL          # -1  /* mem argument is NULL          */
    enum: SPFGMR_ATIMES_FAIL_UNREC # -2  /* atimes returned failure flag  */
    enum: SPFGMR_PSOLVE_FAIL_UNREC # -3  /* psolve failed unrecoverably   */
    enum: SPFGMR_GS_FAIL           # -4  /* Gram-Schmidt routine faiuled  */
    enum: SPFGMR_QRSOL_FAIL        # -5  /* QRsol found singular R        */
    enum: SPFGMR_PSET_FAIL_UNREC   # -6  /* pset failed unrecoverably     */

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

# We don't use SuperLUMT - "slu_mt_ddefs.h" required
#cdef extern from "sundials/sundials_superlumt_impl.h":
#    ctypedef struct SLUMTDataRec:
#        SuperMatrix *s_A, *s_AC, *s_L, *s_U, *s_B
#        Gstat_t *Gstat
#        int *perm_r, *perm_c
#        int num_threads
#        double diag_pivot_thresh
#        superlumt_options_t *superlumt_options
#
#        int s_ordering
#
#    ctypedef SLUMTDataRec *SLUMTData

cdef extern from "sundials/sundials_nonlinearsolver.h":

    struct _generic_SUNNonlinearSolver_Ops:
        pass
    struct _generic_SUNNonlinearSolver:
        pass
    ctypedef _generic_SUNNonlinearSolver_Ops *SUNNonlinearSolver_Ops
    ctypedef _generic_SUNNonlinearSolver *SUNNonlinearSolver

    ctypedef int (*SUNNonlinSolSysFn)(N_Vector y, N_Vector F, void* mem)
    ctypedef int (*SUNNonlinSolLSetupFn)(N_Vector y, N_Vector F, 
                                         booleantype jbad,
                                         booleantype* jcur, void* mem)
    ctypedef int (*SUNNonlinSolLSolveFn)(N_Vector y, N_Vector b, void* mem)
    # rename reserved del into del_t for python!
    ctypedef int (*SUNNonlinSolConvTestFn)(SUNNonlinearSolver NLS, N_Vector y,
                                          N_Vector del_t, realtype tol, 
                                          N_Vector ewt, void* mem)

    cdef enum SUNNonlinearSolver_Type:
        SUNNONLINEARSOLVER_ROOTFIND,
        SUNNONLINEARSOLVER_FIXEDPOINT

    struct _generic_SUNNonlinearSolver_Ops:
        SUNNonlinearSolver_Type (*gettype)(SUNNonlinearSolver)
        int (*initialize)(SUNNonlinearSolver)
        int (*setup)(SUNNonlinearSolver, N_Vector, void*)
        int (*solve)(SUNNonlinearSolver, N_Vector, N_Vector, N_Vector, realtype,
                     booleantype, void*)
        int (*free)(SUNNonlinearSolver)
        int (*setsysfn)(SUNNonlinearSolver, SUNNonlinSolSysFn)
        int (*setlsetupfn)(SUNNonlinearSolver, SUNNonlinSolLSetupFn)
        int (*setlsolvefn)(SUNNonlinearSolver, SUNNonlinSolLSolveFn)
        int (*setctestfn)(SUNNonlinearSolver, SUNNonlinSolConvTestFn)
        int (*setmaxiters)(SUNNonlinearSolver, int)
        int (*getnumiters)(SUNNonlinearSolver, long int*)
        int (*getcuriter)(SUNNonlinearSolver, int*)
        int (*getnumconvfails)(SUNNonlinearSolver, long int*)

    #struct _generic_SUNNonlinearSolver :
    #    void *content
    #    struct _generic_SUNNonlinearSolver_Ops *ops

    SUNNonlinearSolver_Type SUNNonlinSolGetType(SUNNonlinearSolver NLS)

    int SUNNonlinSolInitialize(SUNNonlinearSolver NLS)
    int SUNNonlinSolSetup(SUNNonlinearSolver NLS, N_Vector y, void* mem)
    int SUNNonlinSolSolve(SUNNonlinearSolver NLS, N_Vector y0, N_Vector y,
                          N_Vector w, realtype tol,
                          booleantype callLSetup, void *mem)
    int SUNNonlinSolFree(SUNNonlinearSolver NLS)

    int SUNNonlinSolSetSysFn(SUNNonlinearSolver NLS,  SUNNonlinSolSysFn SysFn)
    int SUNNonlinSolSetLSetupFn(SUNNonlinearSolver NLS,
                                SUNNonlinSolLSetupFn SetupFn)
    int SUNNonlinSolSetLSolveFn(SUNNonlinearSolver NLS,
                                SUNNonlinSolLSolveFn SolveFn)
    int SUNNonlinSolSetConvTestFn(SUNNonlinearSolver NLS,
                                  SUNNonlinSolConvTestFn CTestFn)
    int SUNNonlinSolSetMaxIters(SUNNonlinearSolver NLS, int maxiters)

    int SUNNonlinSolGetNumIters(SUNNonlinearSolver NLS, long int *niters)
    int SUNNonlinSolGetCurIter(SUNNonlinearSolver NLS, int *iter)
    int SUNNonlinSolGetNumConvFails(SUNNonlinearSolver NLS,
                                    long int *nconvfails)


    enum: SUN_NLS_SUCCESS         #  0
    enum: SUN_NLS_CONTINUE        # +1
    enum: SUN_NLS_CONV_RECVR      # +2
    
    enum: SUN_NLS_MEM_NULL        # -1
    enum: SUN_NLS_MEM_FAIL        # -2
    enum: SUN_NLS_ILL_INPUT       # -3
    enum: SUN_NLS_VECTOROP_ERR    # -4
