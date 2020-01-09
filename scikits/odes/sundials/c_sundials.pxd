from libc.stdio cimport FILE

cdef extern from "sundials/sundials_types.h":
    ctypedef float realtype
    ctypedef unsigned int booleantype
    ctypedef long sunindextype

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
    ctypedef _generic_N_Vector *N_Vector
    ctypedef _generic_N_Vector_Ops *N_Vector_Ops
    ctypedef N_Vector *N_Vector_S

    struct _generic_N_Vector_Ops:
        N_Vector_ID (*nvgetvectorid)(N_Vector)
        N_Vector    (*nvclone)(N_Vector)
        N_Vector    (*nvcloneempty)(N_Vector)
        void        (*nvdestroy)(N_Vector)
        void        (*nvspace)(N_Vector, sunindextype *, sunindextype *)
        realtype*   (*nvgetarraypointer)(N_Vector)
        void        (*nvsetarraypointer)(realtype *, N_Vector)
        void*       (*nvgetcommunicator)(N_Vector)
        sunindextype (*nvgetlength)(N_Vector)
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
        int (*nvlinearcombination)(int, realtype*, N_Vector*, N_Vector)
        int (*nvscaleaddmulti)(int, realtype*, N_Vector, N_Vector*, N_Vector*)
        int (*nvdotprodmulti)(int, N_Vector, N_Vector*, realtype*)
    
        int (*nvlinearsumvectorarray)(int, realtype, N_Vector*, realtype, 
                                      N_Vector*, N_Vector*)
        int (*nvscalevectorarray)(int, realtype*, N_Vector*, N_Vector*)
        int (*nvconstvectorarray)(int, realtype, N_Vector*)
        int (*nvwrmsnormvectorarray)(int, N_Vector*, N_Vector*, realtype*)
        int (*nvwrmsnormmaskvectorarray)(int, N_Vector*, N_Vector*, N_Vector,
                                         realtype*)
        int (*nvscaleaddmultivectorarray)(int, int, realtype*, N_Vector*, 
                                          N_Vector**, N_Vector**)
        int (*nvlinearcombinationvectorarray)(int, int, realtype*, N_Vector**,
                                              N_Vector*)
    
        realtype (*nvdotprodlocal)(N_Vector, N_Vector)
        realtype (*nvmaxnormlocal)(N_Vector)
        realtype (*nvminlocal)(N_Vector)
        realtype (*nvl1normlocal)(N_Vector)
        booleantype (*nvinvtestlocal)(N_Vector, N_Vector)
        booleantype (*nvconstrmasklocal)(N_Vector, N_Vector, N_Vector)
        realtype (*nvminquotientlocal)(N_Vector, N_Vector)
        realtype (*nvwsqrsumlocal)(N_Vector, N_Vector)
        realtype (*nvwsqrsummasklocal)(N_Vector, N_Vector, N_Vector)

    struct _generic_N_Vector:
        void *content
        N_Vector_Ops ops


    # * FUNCTIONS *
    N_Vector N_VNewEmpty()
    void N_VFreeEmpty(N_Vector v)
    int N_VCopyOps(N_Vector w, N_Vector v)

    N_Vector_ID N_VGetVectorID(N_Vector w)
    N_Vector N_VClone(N_Vector w)
    N_Vector N_VCloneEmpty(N_Vector w)
    void N_VDestroy(N_Vector v)
    void N_VSpace(N_Vector v, sunindextype *lrw, sunindextype *liw)
    realtype *N_VGetArrayPointer(N_Vector v)
    void N_VSetArrayPointer(realtype *v_data, N_Vector v)
    void *N_VGetCommunicator(N_Vector v)
    sunindextype N_VGetLength(N_Vector v)

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
    
    # /* OPTIONAL fused vector operations */
    int N_VLinearCombination(int nvec, realtype* c, N_Vector* X, N_Vector z)

    int N_VScaleAddMulti(int nvec, realtype* a, N_Vector x,
                         N_Vector* Y, N_Vector* Z)

    int N_VDotProdMulti(int nvec, N_Vector x, N_Vector* Y, realtype* dotprods)

    #/* OPTIONAL vector array operations */
    int N_VLinearSumVectorArray(int nvec, realtype a, N_Vector* X,
                                realtype b, N_Vector* Y, N_Vector* Z)

    int N_VScaleVectorArray(int nvec, realtype* c, N_Vector* X, N_Vector* Z)

    int N_VConstVectorArray(int nvec, realtype c, N_Vector* Z)

    int N_VWrmsNormVectorArray(int nvec, N_Vector* X, N_Vector* W, 
                               realtype* nrm)

    int N_VWrmsNormMaskVectorArray(int nvec, N_Vector* X, N_Vector* W, 
                                   N_Vector id, realtype* nrm)

    int N_VScaleAddMultiVectorArray(int nvec, int nsum, realtype* a, 
                                    N_Vector* X, N_Vector** Y, N_Vector** Z)

    int N_VLinearCombinationVectorArray(int nvec, int nsum, realtype* c, 
                                        N_Vector** X, N_Vector* Z)

    #/* OPTIONAL local reduction kernels (no parallel communication) */
    realtype N_VDotProdLocal(N_Vector x, N_Vector y)
    realtype N_VMaxNormLocal(N_Vector x)
    realtype N_VMinLocal(N_Vector x)
    realtype N_VL1NormLocal(N_Vector x)
    realtype N_VWSqrSumLocal(N_Vector x, N_Vector w)
    realtype N_VWSqrSumMaskLocal(N_Vector x, N_Vector w, N_Vector id)
    booleantype N_VInvTestLocal(N_Vector x, N_Vector z)
    booleantype N_VConstrMaskLocal(N_Vector c, N_Vector x, N_Vector m)
    realtype N_VMinQuotientLocal(N_Vector num, N_Vector denom)

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
        int          (*scaleadd)(realtype, SUNMatrix, SUNMatrix)
        int          (*scaleaddi)(realtype, SUNMatrix)
        int          (*matvecsetup)(SUNMatrix);
        int          (*matvec)(SUNMatrix, N_Vector, N_Vector)
        int          (*space)(SUNMatrix, long int*, long int*)

    struct _generic_SUNMatrix:
        void *content
        SUNMatrix_Ops ops

    # * FUNCTIONS *
    SUNMatrix SUNMatNewEmpty()
    void SUNMatFreeEmpty(SUNMatrix A)
    int SUNMatCopyOps(SUNMatrix A, SUNMatrix B)
    SUNMatrix_ID SUNMatGetID(SUNMatrix A)
    SUNMatrix SUNMatClone(SUNMatrix A)
    void SUNMatDestroy(SUNMatrix A)
    int SUNMatZero(SUNMatrix A)
    int SUNMatCopy(SUNMatrix A, SUNMatrix B)
    int SUNMatScaleAdd(realtype c, SUNMatrix A, SUNMatrix B)
    int SUNMatScaleAddI(realtype c, SUNMatrix A)
    int SUNMatMatvecSetup(SUNMatrix A)
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

    struct _generic_SUNLinearSolver:
        void *content
        SUNLinearSolver_Ops ops

    SUNLinearSolver SUNLinSolNewEmpty()

    void SUNLinSolFreeEmpty(SUNLinearSolver S)

    SUNLinearSolver_Type SUNLinSolGetType(SUNLinearSolver S)

    SUNLinearSolver_ID SUNLinSolGetID(SUNLinearSolver S)
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

cdef extern from "sundials/sundials_nonlinearsolver.h":

    struct _generic_SUNNonlinearSolver_Ops:
        pass
    struct _generic_SUNNonlinearSolver:
        pass
    ctypedef _generic_SUNNonlinearSolver_Ops *SUNNonlinearSolver_Ops
    ctypedef _generic_SUNNonlinearSolver *SUNNonlinearSolver

    ctypedef int (*SUNNonlinSolSysFn)(N_Vector y, N_Vector F, void* mem)
    ctypedef int (*SUNNonlinSolLSetupFn)(booleantype jbad,
                                         booleantype* jcur, void* mem)
    ctypedef int (*SUNNonlinSolLSolveFn)(N_Vector b, void* mem)
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

    struct _generic_SUNNonlinearSolver :
        void *content
        SUNNonlinearSolver_Ops ops

    SUNNonlinearSolver SUNNonlinSolNewEmpty();
    void SUNNonlinSolFreeEmpty(SUNNonlinearSolver NLS);

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
    enum: SUN_NLS_CONTINUE        # +901
    enum: SUN_NLS_CONV_RECVR      # +902
    
    enum: SUN_NLS_MEM_NULL        # -901
    enum: SUN_NLS_MEM_FAIL        # -902
    enum: SUN_NLS_ILL_INPUT       # -903
    enum: SUN_NLS_VECTOROP_ERR    # -904
    enum: SUN_NLS_EXT_FAIL        # -905 
