from c_sundials cimport *
from libc.stdio cimport FILE

cdef extern from "ida/ida.h":
    enum: IDA_NORMAL
    enum: IDA_ONE_STEP

    #/* icopt */
    enum: IDA_YA_YDP_INIT
    enum: IDA_Y_INIT

    #/* 
    #* ----------------------------------------
    #* IDA return flags 
    #* ----------------------------------------
    #*/

    enum: IDA_SUCCESS
    enum: IDA_TSTOP_RETURN
    enum: IDA_ROOT_RETURN

    enum: IDA_WARNING

    enum: IDA_MEM_NULL
    enum: IDA_ILL_INPUT
    enum: IDA_NO_MALLOC
    enum: IDA_TOO_MUCH_WORK
    enum: IDA_TOO_MUCH_ACC
    enum: IDA_ERR_FAIL
    enum: IDA_CONV_FAIL
    enum: IDA_LINIT_FAIL
    enum: IDA_LSETUP_FAIL
    enum: IDA_LSOLVE_FAIL
    enum: IDA_RES_FAIL
    enum: IDA_CONSTR_FAIL
    enum: IDA_REP_RES_ERR

    enum: IDA_MEM_FAIL

    enum: IDA_BAD_T

    enum: IDA_BAD_EWT
    enum: IDA_FIRST_RES_FAIL
    enum: IDA_LINESEARCH_FAIL
    enum: IDA_NO_RECOVERY

    enum: IDA_RTFUNC_FAIL
    
    ctypedef int (*IDAResFn)(realtype tt, N_Vector yy, N_Vector yp,
                    N_Vector rr, void *user_data)
                    
    ctypedef int (*IDARootFn)(realtype t, N_Vector y, N_Vector yp,
                    realtype *gout, void *user_data)
                    
    ctypedef int (*IDAEwtFn)(N_Vector y, N_Vector ewt, void *user_data)
    
    ctypedef void (*IDAErrHandlerFn)(int error_code, \
                    char *module, char *function, \
                    char *msg, void *user_data)
    
    void *IDACreate()
    
    int IDASetErrHandlerFn(void *ida_mem, IDAErrHandlerFn ehfun, void *eh_data)
    int IDASetErrFile(void *ida_mem, FILE *errfp)
    int IDASetUserData(void *ida_mem, void *user_data)
    int IDASetMaxOrd(void *ida_mem, int maxord)
    int IDASetMaxNumSteps(void *ida_mem, long int mxsteps)
    int IDASetInitStep(void *ida_mem, realtype hin)
    int IDASetMaxStep(void *ida_mem, realtype hmax)
    int IDASetStopTime(void *ida_mem, realtype tstop)
    int IDASetNonlinConvCoef(void *ida_mem, realtype epcon)
    int IDASetMaxErrTestFails(void *ida_mem, int maxnef)
    int IDASetMaxNonlinIters(void *ida_mem, int maxcor)
    int IDASetMaxConvFails(void *ida_mem, int maxncf)
    int IDASetSuppressAlg(void *ida_mem, booleantype suppressalg)
    int IDASetId(void *ida_mem, N_Vector id)
    int IDASetConstraints(void *ida_mem, N_Vector constraints)
    
    

    int IDASetRootDirection(void *ida_mem, int *rootdir)
    int IDASetNoInactiveRootWarn(void *ida_mem)
    
    int IDAInit(void *ida_mem, IDAResFn res,
                realtype t0, N_Vector yy0, N_Vector yp0)
    int IDAReInit(void *ida_mem,
                  realtype t0, N_Vector yy0, N_Vector yp0)
    int IDASStolerances(void *ida_mem, realtype reltol, realtype abstol)
    int IDASVtolerances(void *ida_mem, realtype reltol, N_Vector abstol)
    int IDAWFtolerances(void *ida_mem, IDAEwtFn efun)

    int IDASetNonlinConvCoefIC(void *ida_mem, realtype epiccon)
    int IDASetMaxNumStepsIC(void *ida_mem, int maxnh)
    int IDASetMaxNumJacsIC(void *ida_mem, int maxnj)
    int IDASetMaxNumItersIC(void *ida_mem, int maxnit)
    int IDASetLineSearchOffIC(void *ida_mem, booleantype lsoff)
    int IDASetStepToleranceIC(void *ida_mem, realtype steptol)

    int IDARootInit(void *ida_mem, int nrtfn, IDARootFn g)
    int IDACalcIC(void *ida_mem, int icopt, realtype tout1)
    int IDASolve(void *ida_mem, realtype tout, realtype *tret,
                 N_Vector yret, N_Vector ypret, int itask)
    int IDAGetSolution(void *ida_mem, realtype t, 
                       N_Vector yret, N_Vector ypret)
    
    int IDAGetWorkSpace(void *ida_mem, long int *lenrw, long int *leniw)
    int IDAGetNumSteps(void *ida_mem, long int *nsteps)
    int IDAGetNumResEvals(void *ida_mem, long int *nrevals)
    int IDAGetNumLinSolvSetups(void *ida_mem, long int *nlinsetups)
    int IDAGetNumErrTestFails(void *ida_mem, long int *netfails)
    int IDAGetNumBacktrackOps(void *ida_mem, long int *nbacktr)
    int IDAGetConsistentIC(void *ida_mem, N_Vector yy0_mod, N_Vector yp0_mod)
    int IDAGetLastOrder(void *ida_mem, int *klast)
    int IDAGetCurrentOrder(void *ida_mem, int *kcur)
    int IDAGetActualInitStep(void *ida_mem, realtype *hinused)
    int IDAGetLastStep(void *ida_mem, realtype *hlast)
    int IDAGetCurrentStep(void *ida_mem, realtype *hcur)
    int IDAGetCurrentTime(void *ida_mem, realtype *tcur)
    int IDAGetTolScaleFactor(void *ida_mem, realtype *tolsfact)
    int IDAGetErrWeights(void *ida_mem, N_Vector eweight)
    int IDAGetEstLocalErrors(void *ida_mem, N_Vector ele)
    int IDAGetNumGEvals(void *ida_mem, long int *ngevals)
    int IDAGetRootInfo(void *ida_mem, int *rootsfound)

    int IDAGetIntegratorStats(void *ida_mem, long int *nsteps, 
                              long int *nrevals, long int *nlinsetups, 
                              long int *netfails, int *qlast, int *qcur, 
                              realtype *hinused, realtype *hlast, realtype *hcur, 
                              realtype *tcur)
                                          
    int IDAGetNumNonlinSolvIters(void *ida_mem, long int *nniters)
    int IDAGetNumNonlinSolvConvFails(void *ida_mem, long int *nncfails)
    int IDAGetNonlinSolvStats(void *ida_mem, long int *nniters, 
                              long int *nncfails)
    char *IDAGetReturnFlagName(int flag)
    void IDAFree(void **ida_mem)

cdef extern from "ida/ida_direct.h":
    enum: IDADLS_SUCCESS 
    enum: IDADLS_MEM_NULL 
    enum: IDADLS_LMEM_NULL 
    enum: IDADLS_ILL_INPUT
    enum: IDADLS_MEM_FAIL

    #/* Additional last_flag values */
    enum: IDADLS_JACFUNC_UNRECVR
    enum: IDADLS_JACFUNC_RECVR

    ctypedef int (*IDADlsDenseJacFn)(int N, realtype t, realtype c_j,
                                     N_Vector y, N_Vector yp, N_Vector r, 
                                     DlsMat Jac, void *user_data,
                                     N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)

    ctypedef int (*IDADlsBandJacFn)(int N, int mupper, int mlower,
                                    realtype t, realtype c_j, 
                                    N_Vector y, N_Vector yp, N_Vector r,
                                    DlsMat Jac, void *user_data,
                                    N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
    ctypedef int (*IDADlsBandJacFn)(int N, int mupper, int mlower,
                                    realtype t, realtype c_j, 
                                    N_Vector y, N_Vector yp, N_Vector r,
                                    DlsMat Jac, void *user_data,
                                    N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
    int IDADlsSetDenseJacFn(void *ida_mem, IDADlsDenseJacFn jac)
    int IDADlsSetBandJacFn(void *ida_mem, IDADlsBandJacFn jac)
    int IDADlsGetWorkSpace(void *ida_mem, long int *lenrwLS, long int *leniwLS)
    int IDADlsGetNumJacEvals(void *ida_mem, long int *njevals)
    int IDADlsGetNumResEvals(void *ida_mem, long int *nfevalsLS)
    int IDADlsGetLastFlag(void *ida_mem, int *flag)
    char *IDADlsGetReturnFlagName(int flag)

cdef extern from "ida/ida_band.h":
    int IDABand(void *ida_mem, int Neq, int mupper, int mlower)

cdef extern from "ida/ida_dense.h":
     int IDADense(void *ida_mem, int Neq)

cdef extern from "ida/ida_lapack.h":
     int IDALapackDense(void *ida_mem, int N)
     int IDALapackBand(void *ida_mem, int N, int mupper, int mlower)
     
cdef extern from "ida/ida_spils.h":
    ctypedef int (*IDASpilsPrecSetupFn)(realtype tt,
                                         N_Vector yy, N_Vector yp, N_Vector rr,
                                         realtype c_j, void *user_data,
                                         N_Vector tmp1, N_Vector tmp2,
                                         N_Vector tmp3)
    ctypedef int (*IDASpilsPrecSolveFn)(realtype tt,
                                        N_Vector yy, N_Vector yp, N_Vector rr,
                                        N_Vector rvec, N_Vector zvec,
                                        realtype c_j, realtype delta, void *user_data,
                                        N_Vector tmp)
    ctypedef int (*IDASpilsJacTimesVecFn)(realtype tt,
                                         N_Vector yy, N_Vector yp, N_Vector rr,
                                         N_Vector v, N_Vector Jv,
                                         realtype c_j, void *user_data,
                                         N_Vector tmp1, N_Vector tmp2)     
    int IDASpilsSetPreconditioner(void *ida_mem,
                                              IDASpilsPrecSetupFn pset, 
                                              IDASpilsPrecSolveFn psolve)
    int IDASpilsSetJacTimesVecFn(void *ida_mem,
                                             IDASpilsJacTimesVecFn jtv)

    int IDASpilsSetGSType(void *ida_mem, int gstype)
    int IDASpilsSetMaxRestarts(void *ida_mem, int maxrs)
    int IDASpilsSetMaxl(void *ida_mem, int maxl)
    int IDASpilsSetEpsLin(void *ida_mem, realtype eplifac)
    int IDASpilsSetIncrementFactor(void *ida_mem, realtype dqincfac)
    int IDASpilsGetWorkSpace(void *ida_mem, long int *lenrwLS, long int *leniwLS)
    int IDASpilsGetNumPrecEvals(void *ida_mem, long int *npevals)
    int IDASpilsGetNumPrecSolves(void *ida_mem, long int *npsolves)
    int IDASpilsGetNumLinIters(void *ida_mem, long int *nliters)
    int IDASpilsGetNumConvFails(void *ida_mem, long int *nlcfails)
    int IDASpilsGetNumJtimesEvals(void *ida_mem, long int *njvevals)
    int IDASpilsGetNumResEvals(void *ida_mem, long int *nrevalsLS) 
    int IDASpilsGetLastFlag(void *ida_mem, int *flag)
    char *IDASpilsGetReturnFlagName(int flag)
    
cdef extern from "ida/ida_spgmr.h":
     int IDASpgmr(void *ida_mem, int maxl)

cdef extern from "ida/ida_spbcgs.h":     
     int IDASpbcg(void *ida_mem, int maxl)

cdef extern from "ida/ida_sptfqmr.h":  
     int IDASptfqmr(void *ida_mem, int maxl)
