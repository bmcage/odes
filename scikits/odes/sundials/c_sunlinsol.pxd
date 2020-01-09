from .c_sundials cimport *
from .c_sunmatrix cimport *

include "sundials_config.pxi"

cdef extern from "sunlinsol/sunlinsol_dense.h":

    struct _SUNLinearSolverContent_Dense:
        sunindextype N
        sunindextype *pivots
        long int last_flag

    ctypedef _SUNLinearSolverContent_Dense *SUNLinearSolverContent_Dense
    
    SUNLinearSolver SUNLinSol_Dense(N_Vector y, SUNMatrix A)
    SUNLinearSolver SUNDenseLinearSolver(N_Vector y, SUNMatrix A) #deprecated

    SUNLinearSolver_Type SUNLinSolGetType_Dense(SUNLinearSolver S)
    SUNLinearSolver_ID SUNLinSolGetID_Dense(SUNLinearSolver S)
    int SUNLinSolInitialize_Dense(SUNLinearSolver S)
    int SUNLinSolSetup_Dense(SUNLinearSolver S, SUNMatrix A)
    int SUNLinSolSolve_Dense(SUNLinearSolver S, SUNMatrix A,
                             N_Vector x, N_Vector b, realtype tol)
    sunindextype SUNLinSolLastFlag_Dense(SUNLinearSolver S)
    int SUNLinSolSpace_Dense(SUNLinearSolver S, long int *lenrwLS,
                             long int *leniwLS)
    int SUNLinSolFree_Dense(SUNLinearSolver S)

cdef extern from "sunlinsol/sunlinsol_band.h":

    struct _SUNLinearSolverContent_Band:
        sunindextype N
        sunindextype *pivots
        long int last_flag

    ctypedef _SUNLinearSolverContent_Band *SUNLinearSolverContent_Band

    SUNLinearSolver SUNLinSol_Band(N_Vector y, SUNMatrix A)
    SUNLinearSolver SUNBandLinearSolver(N_Vector y, SUNMatrix A) # deprecated

    SUNLinearSolver_Type SUNLinSolGetType_Band(SUNLinearSolver S)
    SUNLinearSolver_ID SUNLinSolGetID_Band(SUNLinearSolver S)
    int SUNLinSolInitialize_Band(SUNLinearSolver S)
    int SUNLinSolSetup_Band(SUNLinearSolver S, SUNMatrix A)
    int SUNLinSolSolve_Band(SUNLinearSolver S, SUNMatrix A,
                            N_Vector x, N_Vector b, realtype tol)
    sunindextype SUNLinSolLastFlag_Band(SUNLinearSolver S)
    int SUNLinSolSpace_Band(SUNLinearSolver S, long int *lenrwLS,
                            long int *leniwLS)
    int SUNLinSolFree_Band(SUNLinearSolver S)

# We don't support KLU for now
#cdef extern from "sunlinsol/sunlinsol_klu.h":

IF SUNDIALS_BLAS_LAPACK:
    cdef extern from "sunlinsol/sunlinsol_lapackdense.h":

        struct _SUNLinearSolverContent_LapackDense:
            sunindextype N
            sunindextype *pivots
            long int last_flag

        ctypedef _SUNLinearSolverContent_LapackDense *SUNLinearSolverContent_LapackDense

        SUNLinearSolver SUNLinSol_LapackDense(N_Vector y, SUNMatrix A)
        SUNLinearSolver SUNLapackDense(N_Vector y, SUNMatrix A) #deprecated

        SUNLinearSolver_Type SUNLinSolGetType_LapackDense(SUNLinearSolver S)
        SUNLinearSolver_ID SUNLinSolGetID_LapackDense(SUNLinearSolver S)
        int SUNLinSolInitialize_LapackDense(SUNLinearSolver S)
        int SUNLinSolSetup_LapackDense(SUNLinearSolver S, SUNMatrix A)
        int SUNLinSolSolve_LapackDense(SUNLinearSolver S, SUNMatrix A,
                                       N_Vector x, N_Vector b, realtype tol)
        sunindextype SUNLinSolLastFlag_LapackDense(SUNLinearSolver S)
        int SUNLinSolSpace_LapackDense(SUNLinearSolver S, long int *lenrwLS,
                                       long int *leniwLS)
        int SUNLinSolFree_LapackDense(SUNLinearSolver S)


    cdef extern from "sunlinsol/sunlinsol_lapackband.h":

        struct _SUNLinearSolverContent_LapackBand:
            sunindextype N
            sunindextype *pivots
            long int last_flag

        ctypedef _SUNLinearSolverContent_LapackBand *SUNLinearSolverContent_LapackBand

        SUNLinearSolver SUNLinSol_LapackBand(N_Vector y, SUNMatrix A)
        SUNLinearSolver SUNLapackBand(N_Vector y, SUNMatrix A) # deprecated

        SUNLinearSolver_Type SUNLinSolGetType_LapackBand(SUNLinearSolver S)
        SUNLinearSolver_ID SUNLinSolGetID_LapackBand(SUNLinearSolver S)
        int SUNLinSolInitialize_LapackBand(SUNLinearSolver S)
        int SUNLinSolSetup_LapackBand(SUNLinearSolver S, SUNMatrix A)
        int SUNLinSolSolve_LapackBand(SUNLinearSolver S, SUNMatrix A,
                                      N_Vector x, N_Vector b, realtype tol)
        sunindextype SUNLinSolLastFlag_LapackBand(SUNLinearSolver S)
        int SUNLinSolSpace_LapackBand(SUNLinearSolver S, long int *lenrwLS,
                                      long int *leniwLS)
        int SUNLinSolFree_LapackBand(SUNLinearSolver S)


cdef extern from "sunlinsol/sunlinsol_pcg.h":

    struct _SUNLinearSolverContent_PCG:
        int maxl
        int pretype
        int numiters
        realtype resnorm
        long int last_flag

        ATimesFn ATimes
        void* ATData
        PSetupFn Psetup
        PSolveFn Psolve
        void* PData

        N_Vector s
        N_Vector r
        N_Vector p
        N_Vector z
        N_Vector Ap

    ctypedef _SUNLinearSolverContent_PCG *SUNLinearSolverContent_PCG

    SUNLinearSolver SUNLinSol_PCG(N_Vector y, int pretype, int maxl)
    int SUNLinSol_PCGSetPrecType(SUNLinearSolver S, int pretype)
    int SUNLinSol_PCGSetMaxl(SUNLinearSolver S, int maxl)
    SUNLinearSolver SUNPCG(N_Vector y, int pretype, int maxl) #deprecated
    int SUNPCGSetPrecType(SUNLinearSolver S, int pretype)     #deprecated
    int SUNPCGSetMaxl(SUNLinearSolver S, int maxl)            #deprecated

    SUNLinearSolver_Type SUNLinSolGetType_PCG(SUNLinearSolver S)
    SUNLinearSolver_ID SUNLinSolGetID_PCG(SUNLinearSolver S)
    int SUNLinSolInitialize_PCG(SUNLinearSolver S)
    int SUNLinSolSetATimes_PCG(SUNLinearSolver S, void* A_data, ATimesFn ATimes)
    int SUNLinSolSetPreconditioner_PCG(SUNLinearSolver S, void* P_data,
                                       PSetupFn Pset, PSolveFn Psol)
    int SUNLinSolSetScalingVectors_PCG(SUNLinearSolver S, N_Vector s,
                                       N_Vector nul)
    int SUNLinSolSetup_PCG(SUNLinearSolver S, SUNMatrix nul)
    int SUNLinSolSolve_PCG(SUNLinearSolver S, SUNMatrix nul,
                           N_Vector x, N_Vector b, realtype tol)
    int SUNLinSolNumIters_PCG(SUNLinearSolver S)
    realtype SUNLinSolResNorm_PCG(SUNLinearSolver S)
    N_Vector SUNLinSolResid_PCG(SUNLinearSolver S)
    sunindextype SUNLinSolLastFlag_PCG(SUNLinearSolver S)
    int SUNLinSolSpace_PCG(SUNLinearSolver S, long int *lenrwLS,
                           long int *leniwLS)
    int SUNLinSolFree_PCG(SUNLinearSolver S)


cdef extern from "sunlinsol/sunlinsol_spbcgs.h":

    struct _SUNLinearSolverContent_SPBCGS:
        int maxl
        int pretype
        int numiters
        realtype resnorm
        long int last_flag

        ATimesFn ATimes
        void* ATData
        PSetupFn Psetup
        PSolveFn Psolve
        void* PData

        N_Vector s1
        N_Vector s2
        N_Vector r
        N_Vector r_star
        N_Vector p
        N_Vector q
        N_Vector u
        N_Vector Ap
        N_Vector vtemp

    ctypedef _SUNLinearSolverContent_SPBCGS *SUNLinearSolverContent_SPBCGS

    SUNLinearSolver SUNLinSol_SPBCGS(N_Vector y, int pretype, int maxl)
    int SUNLinSol_SPBCGSSetPrecType(SUNLinearSolver S, int pretype)
    int SUNLinSol_SPBCGSSetMaxl(SUNLinearSolver S, int maxl)
    SUNLinearSolver SUNSPBCGS(N_Vector y, int pretype, int maxl) # deprecated
    int SUNSPBCGSSetPrecType(SUNLinearSolver S, int pretype) # deprecated
    int SUNSPBCGSSetMaxl(SUNLinearSolver S, int maxl) # deprecated

    SUNLinearSolver_Type SUNLinSolGetType_SPBCGS(SUNLinearSolver S)
    SUNLinearSolver_ID SUNLinSolGetID_SPBCGS(SUNLinearSolver S)
    int SUNLinSolInitialize_SPBCGS(SUNLinearSolver S)
    int SUNLinSolSetATimes_SPBCGS(SUNLinearSolver S, void* A_data,
                                                  ATimesFn ATimes)
    int SUNLinSolSetPreconditioner_SPBCGS(SUNLinearSolver S,
                                                          void* P_data,
                                                          PSetupFn Pset,
                                                          PSolveFn Psol)
    int SUNLinSolSetScalingVectors_SPBCGS(SUNLinearSolver S,
                                                          N_Vector s1,
                                                          N_Vector s2)
    int SUNLinSolSetup_SPBCGS(SUNLinearSolver S, SUNMatrix A)
    int SUNLinSolSolve_SPBCGS(SUNLinearSolver S, SUNMatrix A,
                                              N_Vector x, N_Vector b, realtype tol)
    int SUNLinSolNumIters_SPBCGS(SUNLinearSolver S)
    realtype SUNLinSolResNorm_SPBCGS(SUNLinearSolver S)
    N_Vector SUNLinSolResid_SPBCGS(SUNLinearSolver S)
    sunindextype SUNLinSolLastFlag_SPBCGS(SUNLinearSolver S)
    int SUNLinSolSpace_SPBCGS(SUNLinearSolver S,
                                              long int *lenrwLS,
                                              long int *leniwLS)
    int SUNLinSolFree_SPBCGS(SUNLinearSolver S)


cdef extern from "sunlinsol/sunlinsol_spfgmr.h":

    struct _SUNLinearSolverContent_SPFGMR:
        int maxl
        int pretype
        int gstype
        int max_restarts
        int numiters
        realtype resnorm
        long int last_flag

        ATimesFn ATimes
        void* ATData
        PSetupFn Psetup
        PSolveFn Psolve
        void* PData

        N_Vector s1
        N_Vector s2
        N_Vector *V
        N_Vector *Z
        realtype **Hes
        realtype *givens
        N_Vector xcor
        realtype *yg
        N_Vector vtemp

        realtype *cv
        N_Vector *Xv

    ctypedef _SUNLinearSolverContent_SPFGMR *SUNLinearSolverContent_SPFGMR

    SUNLinearSolver SUNLinSol_SPFGMR(N_Vector y, int pretype, int maxl)
    int SUNLinSol_SPFGMRSetPrecType(SUNLinearSolver S, int pretype)
    int SUNLinSol_SPFGMRSetGSType(SUNLinearSolver S, int gstype)
    int SUNLinSol_SPFGMRSetMaxRestarts(SUNLinearSolver S, int maxrs)
    SUNLinearSolver SUNSPFGMR(N_Vector y, int pretype, int maxl) # deprecated
    int SUNSPFGMRSetPrecType(SUNLinearSolver S, int pretype) # deprecated
    int SUNSPFGMRSetGSType(SUNLinearSolver S, int gstype) # deprecated
    int SUNSPFGMRSetMaxRestarts(SUNLinearSolver S, int maxrs) # deprecated

    SUNLinearSolver_Type SUNLinSolGetType_SPFGMR(SUNLinearSolver S)
    SUNLinearSolver_ID SUNLinSolGetID_SPFGMR(SUNLinearSolver S)
    int SUNLinSolInitialize_SPFGMR(SUNLinearSolver S)
    int SUNLinSolSetATimes_SPFGMR(SUNLinearSolver S, void* A_data,
                                                  ATimesFn ATimes)
    int SUNLinSolSetPreconditioner_SPFGMR(SUNLinearSolver S,
                                                          void* P_data,
                                                          PSetupFn Pset,
                                                          PSolveFn Psol)
    int SUNLinSolSetScalingVectors_SPFGMR(SUNLinearSolver S,
                                                          N_Vector s1,
                                                          N_Vector s2)
    int SUNLinSolSetup_SPFGMR(SUNLinearSolver S, SUNMatrix A)
    int SUNLinSolSolve_SPFGMR(SUNLinearSolver S, SUNMatrix A,
                                              N_Vector x, N_Vector b, realtype tol)
    int SUNLinSolNumIters_SPFGMR(SUNLinearSolver S)
    realtype SUNLinSolResNorm_SPFGMR(SUNLinearSolver S)
    N_Vector SUNLinSolResid_SPFGMR(SUNLinearSolver S)
    sunindextype SUNLinSolLastFlag_SPFGMR(SUNLinearSolver S)
    int SUNLinSolSpace_SPFGMR(SUNLinearSolver S, long int *lenrwLS,
                              long int *leniwLS)
    int SUNLinSolFree_SPFGMR(SUNLinearSolver S)


cdef extern from "sunlinsol/sunlinsol_spgmr.h":

    struct _SUNLinearSolverContent_SPGMR:
        int maxl
        int pretype
        int gstype
        int max_restarts
        int numiters
        realtype resnorm
        long int last_flag

        ATimesFn ATimes
        void* ATData
        PSetupFn Psetup
        PSolveFn Psolve
        void* PData

        N_Vector s1
        N_Vector s2
        N_Vector *V
        realtype **Hes
        realtype *givens
        N_Vector xcor
        realtype *yg
        N_Vector vtemp

        realtype *cv
        N_Vector *Xv

    ctypedef _SUNLinearSolverContent_SPGMR *SUNLinearSolverContent_SPGMR

    SUNLinearSolver SUNLinSol_SPGMR(N_Vector y, int pretype, int maxl)
    int SUNLinSol_SPGMRSetPrecType(SUNLinearSolver S, int pretype)
    int SUNLinSol_SPGMRSetGSType(SUNLinearSolver S, int gstype)
    int SUNLinSol_SPGMRSetMaxRestarts(SUNLinearSolver S, int maxrs)
                                                  
    SUNLinearSolver SUNSPGMR(N_Vector y, int pretype, int maxl) # deprecated
    int SUNSPGMRSetPrecType(SUNLinearSolver S, int pretype) # deprecated
    int SUNSPGMRSetGSType(SUNLinearSolver S, int gstype) # deprecated
    int SUNSPGMRSetMaxRestarts(SUNLinearSolver S, int maxrs) # deprecated

    SUNLinearSolver_Type SUNLinSolGetType_SPGMR(SUNLinearSolver S)
    SUNLinearSolver_ID SUNLinSolGetID_SPGMR(SUNLinearSolver S)
    int SUNLinSolInitialize_SPGMR(SUNLinearSolver S)
    int SUNLinSolSetATimes_SPGMR(SUNLinearSolver S, void* A_data,
                                 ATimesFn ATimes)
    int SUNLinSolSetPreconditioner_SPGMR(SUNLinearSolver S, void* P_data,
                                         PSetupFn Pset, PSolveFn Psol)
    int SUNLinSolSetScalingVectors_SPGMR(SUNLinearSolver S, N_Vector s1,
                                         N_Vector s2)
    int SUNLinSolSetup_SPGMR(SUNLinearSolver S, SUNMatrix A)
    int SUNLinSolSolve_SPGMR(SUNLinearSolver S, SUNMatrix A,
                             N_Vector x, N_Vector b, realtype tol)
    int SUNLinSolNumIters_SPGMR(SUNLinearSolver S)
    realtype SUNLinSolResNorm_SPGMR(SUNLinearSolver S)
    N_Vector SUNLinSolResid_SPGMR(SUNLinearSolver S)
    sunindextype SUNLinSolLastFlag_SPGMR(SUNLinearSolver S)
    int SUNLinSolSpace_SPGMR(SUNLinearSolver S, long int *lenrwLS,
                             long int *leniwLS)
    int SUNLinSolFree_SPGMR(SUNLinearSolver S)


cdef extern from "sunlinsol/sunlinsol_sptfqmr.h":

    struct _SUNLinearSolverContent_SPTFQMR:
        int maxl
        int pretype
        int numiters
        realtype resnorm
        long int last_flag

        ATimesFn ATimes
        void* ATData
        PSetupFn Psetup
        PSolveFn Psolve
        void* PData

        N_Vector s1
        N_Vector s2
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

    ctypedef _SUNLinearSolverContent_SPTFQMR *SUNLinearSolverContent_SPTFQMR

    SUNLinearSolver SUNLinSol_SPTFQMR(N_Vector y, int pretype, int maxl)
    int SUNLinSol_SPTFQMRSetPrecType(SUNLinearSolver S, int pretype)
    int SUNLinSol_SPTFQMRSetMaxl(SUNLinearSolver S, int maxl)
    SUNLinearSolver SUNSPTFQMR(N_Vector y, int pretype, int maxl) # deprecated
    int SUNSPTFQMRSetPrecType(SUNLinearSolver S, int pretype) # deprecated
    int SUNSPTFQMRSetMaxl(SUNLinearSolver S, int maxl) # deprecated

    SUNLinearSolver_Type SUNLinSolGetType_SPTFQMR(SUNLinearSolver S)
    SUNLinearSolver_ID SUNLinSolGetID_SPTFQMR(SUNLinearSolver S)
    int SUNLinSolInitialize_SPTFQMR(SUNLinearSolver S)
    int SUNLinSolSetATimes_SPTFQMR(SUNLinearSolver S, void* A_data,
                                   ATimesFn ATimes)
    int SUNLinSolSetPreconditioner_SPTFQMR(SUNLinearSolver S, void* P_data,
                                           PSetupFn Pset, PSolveFn Psol)
    int SUNLinSolSetScalingVectors_SPTFQMR(SUNLinearSolver S, N_Vector s1,
                                           N_Vector s2)
    int SUNLinSolSetup_SPTFQMR(SUNLinearSolver S, SUNMatrix A)
    int SUNLinSolSolve_SPTFQMR(SUNLinearSolver S, SUNMatrix A,
                               N_Vector x, N_Vector b, realtype tol)
    int SUNLinSolNumIters_SPTFQMR(SUNLinearSolver S)
    realtype SUNLinSolResNorm_SPTFQMR(SUNLinearSolver S)
    N_Vector SUNLinSolResid_SPTFQMR(SUNLinearSolver S)
    long int SUNLinSolLastFlag_SPTFQMR(SUNLinearSolver S)
    int SUNLinSolSpace_SPTFQMR(SUNLinearSolver S, long int *lenrwLS,
                               long int *leniwLS)
    int SUNLinSolFree_SPTFQMR(SUNLinearSolver S)


# We don't use SuperLUMT - "slu_mt_ddefs.h" required
#cdef extern from "sunlinsol/sunlinsol_superlumt.h":
#
#    struct _SUNLinearSolverContent_SuperLUMT:
#        long int     last_flag
#        int          first_factorize
#        SuperMatrix  *A, *AC, *L, *U, *B
#        Gstat_t      *Gstat
#        sunindextype *perm_r, *perm_c
#        sunindextype N
#        int          num_threads
#        realtype     diag_pivot_thresh
#        int          ordering
#        superlumt_options_t *options
#
#    ctypedef _SUNLinearSolverContent_SuperLUMT *SUNLinearSolverContent_SuperLUMT
#
#    SUNLinearSolver SUNSuperLUMT(N_Vector y, SUNMatrix A, int num_threads)
#
#    int SUNSuperLUMTSetOrdering(SUNLinearSolver S, int ordering_choice)
#    SUNLinearSolver_Type SUNLinSolGetType_SuperLUMT(SUNLinearSolver S)
#    int SUNLinSolInitialize_SuperLUMT(SUNLinearSolver S)
#    int SUNLinSolSetup_SuperLUMT(SUNLinearSolver S, SUNMatrix A)
#    int SUNLinSolSolve_SuperLUMT(SUNLinearSolver S, SUNMatrix A,
#                                 N_Vector x, N_Vector b, realtype tol)
#    long int SUNLinSolLastFlag_SuperLUMT(SUNLinearSolver S)
#    int SUNLinSolSpace_SuperLUMT(SUNLinearSolver S, long int *lenrwLS,
#                                 long int *leniwLS)
#    int SUNLinSolFree_SuperLUMT(SUNLinearSolver S)

