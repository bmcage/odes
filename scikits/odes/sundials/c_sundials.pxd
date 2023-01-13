from libc.stdio cimport FILE

cdef extern from "sundials/sundials_types.h":
    ctypedef float sunrealtype
    ctypedef unsigned int sunbooleantype
    ctypedef long sunindextype

cdef extern from "sundials/sundials_context.h":
    struct _SUNContext:
        pass
    ctypedef _SUNContext* SUNContext

    int SUNContext_Create(void* comm, SUNContext* ctx)
    # Need to include profiler/logger headers
    #int SUNContext_GetProfiler(SUNContext sunctx, SUNProfiler* profiler)
    #int SUNContext_SetProfiler(SUNContext sunctx, SUNProfiler profiler)
    #int SUNContext_GetLogger(SUNContext sunctx, SUNLogger* logger)
    #int SUNContext_SetLogger(SUNContext sunctx, SUNLogger logger)
    int SUNContext_Free(SUNContext* ctx)

cdef extern from "sundials/sundials_version.h":
    
    #Fill a string with SUNDIALS version information */
    int SUNDIALSGetVersion(char *version, int len)

    # Fills integers with the major, minor, and patch release version numbers and a string with the release label.
    int SUNDIALSGetVersionNumber(int *major, int *minor, int *patch,
                                 char *label, int len);
                                             
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
        SUNDIALS_NVEC_OPENMPDEV,
        SUNDIALS_NVEC_TRILINOS,
        SUNDIALS_NVEC_MANYVECTOR,
        SUNDIALS_NVEC_MPIMANYVECTOR,
        SUNDIALS_NVEC_MPIPLUSX,
        SUNDIALS_NVEC_CUSTOM

    struct _generic_N_Vector_Ops:
        pass
    struct _generic_N_Vector:
        pass
    ctypedef _generic_N_Vector* N_Vector
    ctypedef _generic_N_Vector_Ops* N_Vector_Ops
    ctypedef N_Vector* N_Vector_S

    struct _generic_N_Vector_Ops:
        N_Vector_ID (*nvgetvectorid)(N_Vector)
        N_Vector    (*nvclone)(N_Vector)
        N_Vector    (*nvcloneempty)(N_Vector)
        void        (*nvdestroy)(N_Vector)
        void        (*nvspace)(N_Vector, sunindextype *, sunindextype *)
        sunrealtype*   (*nvgetarraypointer)(N_Vector)
        void        (*nvsetarraypointer)(sunrealtype *, N_Vector)
        void*       (*nvgetcommunicator)(N_Vector)
        sunindextype (*nvgetlength)(N_Vector)
        void        (*nvlinearsum)(sunrealtype, N_Vector, sunrealtype, N_Vector, N_Vector)
        void        (*nvconst)(sunrealtype, N_Vector)
        void        (*nvprod)(N_Vector, N_Vector, N_Vector)
        void        (*nvdiv)(N_Vector, N_Vector, N_Vector)
        void        (*nvscale)(sunrealtype, N_Vector, N_Vector)
        void        (*nvabs)(N_Vector, N_Vector)
        void        (*nvinv)(N_Vector, N_Vector)
        void        (*nvaddconst)(N_Vector, sunrealtype, N_Vector)
        sunrealtype    (*nvdotprod)(N_Vector, N_Vector)
        sunrealtype    (*nvmaxnorm)(N_Vector)
        sunrealtype    (*nvwrmsnorm)(N_Vector, N_Vector)
        sunrealtype    (*nvwrmsnormmask)(N_Vector, N_Vector, N_Vector)
        sunrealtype    (*nvmin)(N_Vector)
        sunrealtype    (*nvwl2norm)(N_Vector, N_Vector)
        sunrealtype    (*nvl1norm)(N_Vector)
        void        (*nvcompare)(sunrealtype, N_Vector, N_Vector)
        sunbooleantype (*nvinvtest)(N_Vector, N_Vector)
        sunbooleantype (*nvconstrmask)(N_Vector, N_Vector, N_Vector)
        sunrealtype    (*nvminquotient)(N_Vector, N_Vector)
        int (*nvlinearcombination)(int, sunrealtype*, N_Vector*, N_Vector)
        int (*nvscaleaddmulti)(int, sunrealtype*, N_Vector, N_Vector*, N_Vector*)
        int (*nvdotprodmulti)(int, N_Vector, N_Vector*, sunrealtype*)
    
        int (*nvlinearsumvectorarray)(int, sunrealtype, N_Vector*, sunrealtype, 
                                      N_Vector*, N_Vector*)
        int (*nvscalevectorarray)(int, sunrealtype*, N_Vector*, N_Vector*)
        int (*nvconstvectorarray)(int, sunrealtype, N_Vector*)
        int (*nvwrmsnormvectorarray)(int, N_Vector*, N_Vector*, sunrealtype*)
        int (*nvwrmsnormmaskvectorarray)(int, N_Vector*, N_Vector*, N_Vector,
                                         sunrealtype*)
        int (*nvscaleaddmultivectorarray)(int, int, sunrealtype*, N_Vector*, 
                                          N_Vector**, N_Vector**)
        int (*nvlinearcombinationvectorarray)(int, int, sunrealtype*, N_Vector**,
                                              N_Vector*)
    
        sunrealtype (*nvdotprodlocal)(N_Vector, N_Vector)
        sunrealtype (*nvmaxnormlocal)(N_Vector)
        sunrealtype (*nvminlocal)(N_Vector)
        sunrealtype (*nvl1normlocal)(N_Vector)
        sunbooleantype (*nvinvtestlocal)(N_Vector, N_Vector)
        sunbooleantype (*nvconstrmasklocal)(N_Vector, N_Vector, N_Vector)
        sunrealtype (*nvminquotientlocal)(N_Vector, N_Vector)
        sunrealtype (*nvwsqrsumlocal)(N_Vector, N_Vector)
        sunrealtype (*nvwsqrsummasklocal)(N_Vector, N_Vector, N_Vector)

    struct _generic_N_Vector:
        void *content
        N_Vector_Ops ops


    # * FUNCTIONS *
    N_Vector N_VNewEmpty(SUNContext sunctx)
    void N_VFreeEmpty(N_Vector v)
    int N_VCopyOps(N_Vector w, N_Vector v)

    N_Vector_ID N_VGetVectorID(N_Vector w)
    N_Vector N_VClone(N_Vector w)
    N_Vector N_VCloneEmpty(N_Vector w)
    void N_VDestroy(N_Vector v)
    void N_VSpace(N_Vector v, sunindextype *lrw, sunindextype *liw)
    sunrealtype *N_VGetArrayPointer(N_Vector v)
    void N_VSetArrayPointer(sunrealtype *v_data, N_Vector v)
    void *N_VGetCommunicator(N_Vector v)
    sunindextype N_VGetLength(N_Vector v)

    void N_VLinearSum(sunrealtype a, N_Vector x, sunrealtype b, N_Vector y, N_Vector z)
    void N_VConst(sunrealtype c, N_Vector z)
    void N_VProd(N_Vector x, N_Vector y, N_Vector z)
    void N_VDiv(N_Vector x, N_Vector y, N_Vector z)
    void N_VScale(sunrealtype c, N_Vector x, N_Vector z)
    void N_VAbs(N_Vector x, N_Vector z)
    void N_VInv(N_Vector x, N_Vector z)
    void N_VAddConst(N_Vector x, sunrealtype b, N_Vector z)
    sunrealtype N_VDotProd(N_Vector x, N_Vector y)
    sunrealtype N_VMaxNorm(N_Vector x)
    sunrealtype N_VWrmsNorm(N_Vector x, N_Vector w)
    sunrealtype N_VWrmsNormMask(N_Vector x, N_Vector w, N_Vector id)
    sunrealtype N_VMin(N_Vector x)
    sunrealtype N_VWL2Norm(N_Vector x, N_Vector w)
    sunrealtype N_VL1Norm(N_Vector x)
    void N_VCompare(sunrealtype c, N_Vector x, N_Vector z)
    sunbooleantype N_VInvTest(N_Vector x, N_Vector z)
    sunbooleantype N_VConstrMask(N_Vector c, N_Vector x, N_Vector m)
    sunrealtype N_VMinQuotient(N_Vector num, N_Vector denom)
    
    # /* OPTIONAL fused vector operations */
    int N_VLinearCombination(int nvec, sunrealtype* c, N_Vector* X, N_Vector z)

    int N_VScaleAddMulti(int nvec, sunrealtype* a, N_Vector x,
                         N_Vector* Y, N_Vector* Z)

    int N_VDotProdMulti(int nvec, N_Vector x, N_Vector* Y, sunrealtype* dotprods)

    #/* OPTIONAL vector array operations */
    int N_VLinearSumVectorArray(int nvec, sunrealtype a, N_Vector* X,
                                sunrealtype b, N_Vector* Y, N_Vector* Z)

    int N_VScaleVectorArray(int nvec, sunrealtype* c, N_Vector* X, N_Vector* Z)

    int N_VConstVectorArray(int nvec, sunrealtype c, N_Vector* Z)

    int N_VWrmsNormVectorArray(int nvec, N_Vector* X, N_Vector* W, 
                               sunrealtype* nrm)

    int N_VWrmsNormMaskVectorArray(int nvec, N_Vector* X, N_Vector* W, 
                                   N_Vector id, sunrealtype* nrm)

    int N_VScaleAddMultiVectorArray(int nvec, int nsum, sunrealtype* a, 
                                    N_Vector* X, N_Vector** Y, N_Vector** Z)

    int N_VLinearCombinationVectorArray(int nvec, int nsum, sunrealtype* c, 
                                        N_Vector** X, N_Vector* Z)

    #/* OPTIONAL local reduction kernels (no parallel communication) */
    sunrealtype N_VDotProdLocal(N_Vector x, N_Vector y)
    sunrealtype N_VMaxNormLocal(N_Vector x)
    sunrealtype N_VMinLocal(N_Vector x)
    sunrealtype N_VL1NormLocal(N_Vector x)
    sunrealtype N_VWSqrSumLocal(N_Vector x, N_Vector w)
    sunrealtype N_VWSqrSumMaskLocal(N_Vector x, N_Vector w, N_Vector id)
    sunbooleantype N_VInvTestLocal(N_Vector x, N_Vector z)
    sunbooleantype N_VConstrMaskLocal(N_Vector c, N_Vector x, N_Vector m)
    sunrealtype N_VMinQuotientLocal(N_Vector num, N_Vector denom)

    # * Additional functions exported by NVECTOR module

    N_Vector* N_VNewVectorArray(int count)
    N_Vector *N_VCloneEmptyVectorArray(int count, N_Vector w)
    N_Vector *N_VCloneVectorArray(int count, N_Vector w)
    void N_VDestroyVectorArray(N_Vector *vs, int count)

    #/* These function are really only for users of the Fortran interface */
    N_Vector N_VGetVecAtIndexVectorArray(N_Vector* vs, int index)
    void N_VSetVecAtIndexVectorArray(N_Vector* vs, int index, N_Vector w)

cdef extern from "sundials/sundials_matrix.h":
    cdef enum SUNMatrix_ID:
        SUNMATRIX_DENSE,
        SUNMATRIX_BAND,
        SUNMATRIX_SPARSE,
        SUNMATRIX_SLUNRLOC,
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
        int          (*scaleadd)(sunrealtype, SUNMatrix, SUNMatrix)
        int          (*scaleaddi)(sunrealtype, SUNMatrix)
        int          (*matvecsetup)(SUNMatrix);
        int          (*matvec)(SUNMatrix, N_Vector, N_Vector)
        int          (*space)(SUNMatrix, long int*, long int*)

    struct _generic_SUNMatrix:
        void *content
        SUNMatrix_Ops ops

    # * FUNCTIONS *
    SUNMatrix SUNMatNewEmpty(SUNContext sunctx)
    void SUNMatFreeEmpty(SUNMatrix A)
    int SUNMatCopyOps(SUNMatrix A, SUNMatrix B)
    SUNMatrix_ID SUNMatGetID(SUNMatrix A)
    SUNMatrix SUNMatClone(SUNMatrix A)
    void SUNMatDestroy(SUNMatrix A)
    int SUNMatZero(SUNMatrix A)
    int SUNMatCopy(SUNMatrix A, SUNMatrix B)
    int SUNMatScaleAdd(sunrealtype c, SUNMatrix A, SUNMatrix B)
    int SUNMatScaleAddI(sunrealtype c, SUNMatrix A)
    int SUNMatMatvecSetup(SUNMatrix A)
    int SUNMatMatvec(SUNMatrix A, N_Vector x, N_Vector y)
    int SUNMatSpace(SUNMatrix A, long int *lenrw, long int *leniw)

cdef extern from "sundials/sundials_iterative.h":
    enum:
        SUN_PREC_NONE
        SUN_PREC_LEFT
        SUN_PREC_RIGHT
        SUN_PREC_BOTH
    enum:
        SUN_MODIFIED_GS = 1
        SUN_CLASSICAL_GS = 2
    ctypedef int (*SUNATimesFn)(void *A_data, N_Vector v, N_Vector z)
    ctypedef int (*SUNPSetupFn)(void *P_data)
    ctypedef int (*SUNPSolveFn)(void *P_data, N_Vector r, N_Vector z,
                             sunrealtype tol, int lr)

    int ModifiedGS(N_Vector *v, sunrealtype **h, int k, int p,
                   sunrealtype *new_vk_norm)
    int ClassicalGS(N_Vector *v, sunrealtype **h, int k, int p,
                    sunrealtype *new_vk_norm, N_Vector temp, sunrealtype *s)
    int QRfact(int n, sunrealtype **h, sunrealtype *q, int job)
    int QRsol(int n, sunrealtype **h, sunrealtype *q, sunrealtype *b)

    enum: SUNMAT_SUCCESS                  #    0  /* function successfull          */
    enum: SUNMAT_ILL_INPUT                # -701  /* illegal function input        */
    enum: SUNMAT_MEM_FAIL                 # -702  /* failed memory access/alloc    */
    enum: SUNMAT_OPERATION_FAIL           # -703  /* a SUNMatrix operation returned nonzero */
    enum: SUNMAT_MATVEC_SETUP_REQUIRED    # -704  /* the SUNMatMatvecSetup routine needs to be called */

cdef extern from "sundials/sundials_linearsolver.h":

    cdef enum SUNLinearSolver_Type:
        SUNLINEARSOLVER_DIRECT,
        SUNLINEARSOLVER_ITERATIVE,
        SUNLINEARSOLVER_MATRIX_ITERATIVE


    cdef enum SUNLinearSolver_ID:
        SUNLINEARSOLVER_BAND,
        SUNLINEARSOLVER_DENSE,
        SUNLINEARSOLVER_KLU,
        SUNLINEARSOLVER_LAPACKBAND,
        SUNLINEARSOLVER_LAPACKDENSE,
        SUNLINEARSOLVER_PCG,
        SUNLINEARSOLVER_SPBCGS,
        SUNLINEARSOLVER_SPFGMR,
        SUNLINEARSOLVER_SPGMR,
        SUNLINEARSOLVER_SPTFQMR,
        SUNLINEARSOLVER_SUPERLUDIST,
        SUNLINEARSOLVER_SUPERLUMT,
        SUNLINEARSOLVER_CUSOLVERSP_BATCHQR,
        SUNLINEARSOLVER_CUSTOM
        
    struct _generic_SUNLinearSolver_Ops:
        pass
    struct _generic_SUNLinearSolver:
        pass
    ctypedef _generic_SUNLinearSolver *SUNLinearSolver
    ctypedef _generic_SUNLinearSolver_Ops *SUNLinearSolver_Ops

    struct _generic_SUNLinearSolver_Ops:
        SUNLinearSolver_Type (*gettype)(SUNLinearSolver)
        SUNLinearSolver_ID   (*getid)(SUNLinearSolver);
        int                  (*setatimes)(SUNLinearSolver, void*, SUNATimesFn)
        int                  (*setpreconditioner)(SUNLinearSolver, void*,
                                                SUNPSetupFn, SUNPSolveFn)
        int                  (*setscalingvectors)(SUNLinearSolver,
                                                N_Vector, N_Vector)
        int                  (*initialize)(SUNLinearSolver)
        int                  (*setup)(SUNLinearSolver, SUNMatrix)
        int                  (*solve)(SUNLinearSolver, SUNMatrix, N_Vector,
                                    N_Vector, sunrealtype)
        int                  (*numiters)(SUNLinearSolver)
        sunrealtype             (*resnorm)(SUNLinearSolver)
        long int             (*lastflag)(SUNLinearSolver)
        int                  (*space)(SUNLinearSolver, long int*, long int*)
        N_Vector             (*resid)(SUNLinearSolver)
        int                  (*free)(SUNLinearSolver)

    struct _generic_SUNLinearSolver:
        void *content
        SUNLinearSolver_Ops ops

    SUNLinearSolver SUNLinSolNewEmpty(SUNContext sunctx)

    void SUNLinSolFreeEmpty(SUNLinearSolver S)

    SUNLinearSolver_Type SUNLinSolGetType(SUNLinearSolver S)

    SUNLinearSolver_ID SUNLinSolGetID(SUNLinearSolver S)
    int SUNLinSolSetATimes(SUNLinearSolver S, void* A_data,
                           SUNATimesFn ATimes)
    int SUNLinSolSetPreconditioner(SUNLinearSolver S, void* P_data,
                                   SUNPSetupFn Pset, SUNPSolveFn Psol)
    int SUNLinSolSetScalingVectors(SUNLinearSolver S, N_Vector s1,
                                   N_Vector s2)
    int SUNLinSolInitialize(SUNLinearSolver S)
    int SUNLinSolSetup(SUNLinearSolver S, SUNMatrix A)
    int SUNLinSolSolve(SUNLinearSolver S, SUNMatrix A, N_Vector x,
                       N_Vector b, sunrealtype tol)
    int SUNLinSolNumIters(SUNLinearSolver S)
    sunrealtype SUNLinSolResNorm(SUNLinearSolver S)
    N_Vector SUNLinSolResid(SUNLinearSolver S)
    long int SUNLinSolLastFlag(SUNLinearSolver S)
    int SUNLinSolSpace(SUNLinearSolver S, long int *lenrwLS,
                                       long int *leniwLS)
    int SUNLinSolFree(SUNLinearSolver S)

    enum: SUNLS_SUCCESS            #   0   /* successful/converged          */
    
    enum: SUNLS_MEM_NULL           # -801   /* mem argument is NULL          */
    enum: SUNLS_ILL_INPUT          # -802   /* illegal function input        */
    enum: SUNLS_MEM_FAIL           # -803   /* failed memory access          */
    enum: SUNLS_ATIMES_FAIL_UNREC  # -804   /* atimes unrecoverable failure  */
    enum: SUNLS_PSET_FAIL_UNREC    # -805   /* pset unrecoverable failure    */
    enum: SUNLS_PSOLVE_FAIL_UNREC  # -806   /* psolve unrecoverable failure  */
    enum: SUNLS_PACKAGE_FAIL_UNREC # -807   /* external package unrec. fail  */
    enum: SUNLS_GS_FAIL            # -808   /* Gram-Schmidt failure          */
    enum: SUNLS_QRSOL_FAIL         # -809   /* QRsol found singular R        */
    enum: SUNLS_VECTOROP_ERR       # -810   /* vector operation error        */

    enum: SUNLS_RES_REDUCED        # 801   /* nonconv. solve, resid reduced */
    enum: SUNLS_CONV_FAIL          # 802   /* nonconvergent solve           */
    enum: SUNLS_ATIMES_FAIL_REC    # 803   /* atimes failed recoverably     */
    enum: SUNLS_PSET_FAIL_REC      # 804   /* pset failed recoverably       */
    enum: SUNLS_PSOLVE_FAIL_REC    # 805   /* psolve failed recoverably     */
    enum: SUNLS_PACKAGE_FAIL_REC   # 806   /* external package recov. fail  */
    enum: SUNLS_QRFACT_FAIL        # 807   /* QRfact found singular matrix  */
    enum: SUNLS_LUFACT_FAIL        # 808   /* LUfact found singular matrix  */

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
        sunrealtype *data
        sunindextype ldata
        sunrealtype **cols

    ctypedef _DlsMat *SUNDlsMat

    SUNDlsMat NewDenseMat(sunindextype M, sunindextype N)
    SUNDlsMat NewBandMat(sunindextype N, sunindextype mu, sunindextype ml, sunindextype smu)
    void DestroyMat(SUNDlsMat A)
    int *NewIntArray(int N)
    sunindextype *NewIndexArray(sunindextype N)
    sunrealtype *NewRealArray(long int N)
    void DestroyArray(void *p)
    void AddIdentity(SUNDlsMat A)
    void SetToZero(SUNDlsMat A)
    void PrintMat(SUNDlsMat A, FILE *outfile)

    # * Exported function prototypes (functions working on sunrealtype**)
    sunrealtype **newDenseMat(sunindextype m, sunindextype n)
    sunrealtype **newBandMat(sunindextype n, sunindextype smu, sunindextype ml)
    void destroyMat(sunrealtype **a)
    int *newIntArray(int n)
    sunindextype *newIndexArray(sunindextype n)
    sunrealtype *newRealArray(sunindextype m)
    void destroyArray(void *v)

cdef extern from "sundials/sundials_band.h":
    sunindextype SUNDlsMat_BandGBTRF(SUNDlsMat A, sunindextype *p)
    sunindextype SUNDlsMat_bandGBTRF(sunrealtype **a, sunindextype n, sunindextype mu, sunindextype ml,
                       sunindextype smu, sunindextype *p)
    void SUNDlsMat_BandGBTRS(SUNDlsMat A, sunindextype *p, sunrealtype *b)
    void SUNDlsMat_bandGBTRS(sunrealtype **a, sunindextype n, sunindextype smu, sunindextype ml,
                   sunindextype *p, sunrealtype *b)
    void SUNDlsMat_BandCopy(SUNDlsMat A, SUNDlsMat B, sunindextype copymu, sunindextype copyml)
    void SUNDlsMat_bandCopy(sunrealtype **a, sunrealtype **b, sunindextype n, sunindextype a_smu,
                  sunindextype b_smu, sunindextype copymu, sunindextype copyml)
    void SUNDlsMat_BandScale(sunrealtype c, SUNDlsMat A)
    void SUNDlsMat_bandScale(sunrealtype c, sunrealtype **a, sunindextype n, sunindextype mu,
                   sunindextype ml, sunindextype smu)
    void SUNDlsMat_bandAddIdentity(sunrealtype **a, sunindextype n, sunindextype smu)
    void SUNDlsMat_BandMatvec(SUNDlsMat A, sunrealtype *x, sunrealtype *y)
    void SUNDlsMat_bandMatvec(sunrealtype **a, sunrealtype *x, sunrealtype *y, sunindextype n,
                    sunindextype mu, sunindextype ml, sunindextype smu)

cdef extern from "sundials/sundials_dense.h":
    sunindextype SUNDlsMat_DenseGETRF(SUNDlsMat A, sunindextype *p)
    void SUNDlsMat_DenseGETRS(SUNDlsMat A, sunindextype *p, sunrealtype *b)

    sunindextype SUNDlsMat_denseGETRF(sunrealtype **a, sunindextype m, sunindextype n, sunindextype *p)
    void SUNDlsMat_denseGETRS(sunrealtype **a, sunindextype n, sunindextype *p, sunrealtype *b)

    sunindextype SUNDlsMat_DensePOTRF(SUNDlsMat A)
    void SUNDlsMat_DensePOTRS(SUNDlsMat A, sunrealtype *b)

    sunindextype SUNDlsMat_densePOTRF(sunrealtype **a, sunindextype m)
    void SUNDlsMat_densePOTRS(sunrealtype **a, sunindextype m, sunrealtype *b)

    int SUNDlsMat_DenseGEQRF(SUNDlsMat A, sunrealtype *beta, sunrealtype *wrk)
    int SUNDlsMat_DenseORMQR(SUNDlsMat A, sunrealtype *beta, sunrealtype *vn, sunrealtype *vm,
                   sunrealtype *wrk)

    int SUNDlsMat_denseGEQRF(sunrealtype **a, sunindextype m, sunindextype n, sunrealtype *beta, sunrealtype *v)
    int SUNDlsMat_denseORMQR(sunrealtype **a, sunindextype m, sunindextype n, sunrealtype *beta,
                   sunrealtype *v, sunrealtype *w, sunrealtype *wrk)
    void SUNDlsMat_DenseCopy(SUNDlsMat A, SUNDlsMat B)
    void SUNDlsMat_denseCopy(sunrealtype **a, sunrealtype **b, sunindextype m, sunindextype n)
    void SUNDlsMat_DenseScale(sunrealtype c, SUNDlsMat A)
    void SUNDlsMat_denseScale(sunrealtype c, sunrealtype **a, sunindextype m, sunindextype n)
    void SUNDlsMat_denseAddIdentity(sunrealtype **a, sunindextype n)
    void SUNDlsMat_DenseMatvec(SUNDlsMat A, sunrealtype *x, sunrealtype *y)
    void SUNDlsMat_denseMatvec(sunrealtype **a, sunrealtype *x, sunrealtype *y, sunindextype m, sunindextype n)

cdef extern from "sundials/sundials_nonlinearsolver.h":

    struct _generic_SUNNonlinearSolver_Ops:
        pass
    struct _generic_SUNNonlinearSolver:
        pass
    ctypedef _generic_SUNNonlinearSolver_Ops *SUNNonlinearSolver_Ops
    ctypedef _generic_SUNNonlinearSolver *SUNNonlinearSolver

    ctypedef int (*SUNNonlinSolSysFn)(N_Vector y, N_Vector F, void* mem)
    ctypedef int (*SUNNonlinSolLSetupFn)(sunbooleantype jbad,
                                         sunbooleantype* jcur, void* mem)
    ctypedef int (*SUNNonlinSolLSolveFn)(N_Vector b, void* mem)
    # rename reserved del into del_t for python!
    ctypedef int (*SUNNonlinSolConvTestFn)(SUNNonlinearSolver NLS, N_Vector y,
                                          N_Vector del_t, sunrealtype tol, 
                                          N_Vector ewt, void* mem)

    cdef enum SUNNonlinearSolver_Type:
        SUNNONLINEARSOLVER_ROOTFIND,
        SUNNONLINEARSOLVER_FIXEDPOINT

    struct _generic_SUNNonlinearSolver_Ops:
        SUNNonlinearSolver_Type (*gettype)(SUNNonlinearSolver)
        int (*initialize)(SUNNonlinearSolver)
        int (*setup)(SUNNonlinearSolver, N_Vector, void*)
        int (*solve)(SUNNonlinearSolver, N_Vector, N_Vector, N_Vector, sunrealtype,
                     sunbooleantype, void*)
        int (*free)(SUNNonlinearSolver)
        int (*setsysfn)(SUNNonlinearSolver, SUNNonlinSolSysFn)
        int (*setlsetupfn)(SUNNonlinearSolver, SUNNonlinSolLSetupFn)
        int (*setlsolvefn)(SUNNonlinearSolver, SUNNonlinSolLSolveFn)
        int (*setctestfn)(SUNNonlinearSolver, SUNNonlinSolConvTestFn)
        int (*setmaxiters)(SUNNonlinearSolver, int)
        int (*getnumiters)(SUNNonlinearSolver, long int*)
        int (*getcuriter)(SUNNonlinearSolver, int*)
        int (*getnumconvfails)(SUNNonlinearSolver, long int*)

    struct _generic_SUNNonlinearSolver :
        void *content
        SUNNonlinearSolver_Ops ops

    SUNNonlinearSolver SUNNonlinSolNewEmpty(SUNContext sunctx);
    void SUNNonlinSolFreeEmpty(SUNNonlinearSolver NLS);

    SUNNonlinearSolver_Type SUNNonlinSolGetType(SUNNonlinearSolver NLS)

    int SUNNonlinSolInitialize(SUNNonlinearSolver NLS)
    int SUNNonlinSolSetup(SUNNonlinearSolver NLS, N_Vector y, void* mem)
    int SUNNonlinSolSolve(SUNNonlinearSolver NLS, N_Vector y0, N_Vector y,
                          N_Vector w, sunrealtype tol,
                          sunbooleantype callLSetup, void *mem)
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
    enum: SUN_NLS_CONTINUE        # +901
    enum: SUN_NLS_CONV_RECVR      # +902
    
    enum: SUN_NLS_MEM_NULL        # -901
    enum: SUN_NLS_MEM_FAIL        # -902
    enum: SUN_NLS_ILL_INPUT       # -903
    enum: SUN_NLS_VECTOROP_ERR    # -904
    enum: SUN_NLS_EXT_FAIL        # -905 
