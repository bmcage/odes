from .c_sundials cimport *
from libc.stdio cimport FILE

cdef extern from "ida/ida.h":
    enum: IDA_NORMAL          # 1
    enum: IDA_ONE_STEP        # 2

    #/* icopt */
    enum: IDA_YA_YDP_INIT     # 1
    enum: IDA_Y_INIT          # 2

    #/*
    #* ----------------------------------------
    #* IDA return flags
    #* ----------------------------------------
    #*/
    enum: IDA_SUCCESS         # 0
    enum: IDA_TSTOP_RETURN    # 1
    enum: IDA_ROOT_RETURN     # 2

    enum: IDA_WARNING         # 99

    enum: IDA_TOO_MUCH_WORK   #-1
    enum: IDA_TOO_MUCH_ACC    #-2
    enum: IDA_ERR_FAIL        #-3
    enum: IDA_CONV_FAIL       #-4

    enum: IDA_LINIT_FAIL      #-5
    enum: IDA_LSETUP_FAIL     #-6
    enum: IDA_LSOLVE_FAIL     #-7
    enum: IDA_RES_FAIL        #-8
    enum: IDA_REP_RES_ERR     #-9
    enum: IDA_RTFUNC_FAIL     #-10
    enum: IDA_CONSTR_FAIL     #-11

    enum: IDA_FIRST_RES_FAIL  #-12
    enum: IDA_LINESEARCH_FAIL #-13
    enum: IDA_NO_RECOVERY     #-14
    enum: IDA_NLS_INIT_FAIL   #-15
    enum: IDA_NLS_SETUP_FAIL  #-16
    enum: IDA_NLS_FAIL        #-17

    enum: IDA_MEM_NULL        #-20
    enum: IDA_MEM_FAIL        #-21
    enum: IDA_ILL_INPUT       #-22
    enum: IDA_NO_MALLOC       #-23
    enum: IDA_BAD_EWT         #-24
    enum: IDA_BAD_K           #-25
    enum: IDA_BAD_T           #-26
    enum: IDA_BAD_DKY         #-27
    enum: IDA_VECTOROP_ERR    #-28
    
    enum: IDA_UNRECOGNIZED_ERROR #-99

    ctypedef int (*IDAResFn)(realtype tt, N_Vector yy, N_Vector yp,
                    N_Vector rr, void *user_data)

    ctypedef int (*IDARootFn)(realtype t, N_Vector y, N_Vector yp,
                    realtype *gout, void *user_data)

    ctypedef int (*IDAEwtFn)(N_Vector y, N_Vector ewt, void *user_data)

    ctypedef void (*IDAErrHandlerFn)(int error_code, const char *module, 
                                    const char *function, char *msg, 
                                    void *user_data)

    void *IDACreate()

    int IDAInit(void *ida_mem, IDAResFn res,
                realtype t0, N_Vector yy0, N_Vector yp0)
    int IDAReInit(void *ida_mem,
                  realtype t0, N_Vector yy0, N_Vector yp0)
    int IDASStolerances(void *ida_mem, realtype reltol, realtype abstol)
    int IDASVtolerances(void *ida_mem, realtype reltol, N_Vector abstol)
    int IDAWFtolerances(void *ida_mem, IDAEwtFn efun)
    int IDACalcIC(void *ida_mem, int icopt, realtype tout1)
    
    int IDASetNonlinConvCoefIC(void *ida_mem, realtype epiccon)
    int IDASetMaxNumStepsIC(void *ida_mem, int maxnh)
    int IDASetMaxNumJacsIC(void *ida_mem, int maxnj)
    int IDASetMaxNumItersIC(void *ida_mem, int maxnit)
    int IDASetLineSearchOffIC(void *ida_mem, booleantype lsoff)
    int IDASetStepToleranceIC(void *ida_mem, realtype steptol)
    int IDASetMaxBacksIC(void *ida_mem, int maxbacks)
    
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

    int IDASetNonlinearSolver(void *ida_mem, SUNNonlinearSolver NLS)

    int IDARootInit(void *ida_mem, int nrtfn, IDARootFn g)
    int IDASetRootDirection(void *ida_mem, int *rootdir)
    int IDASetNoInactiveRootWarn(void *ida_mem)

    int IDASolve(void *ida_mem, realtype tout, realtype *tret,
                 N_Vector yret, N_Vector ypret, int itask)
    
    int IDAComputeY(void *ida_mem, N_Vector ycor, N_Vector y)
    int IDAComputeYp(void *ida_mem, N_Vector ycor, N_Vector yp)
    
    int IDAGetDky(void *ida_mem, realtype t, int k, N_Vector dky)

    int IDAGetWorkSpace(void *ida_mem, long int *lenrw, long int *leniw)
    int IDAGetNumSteps(void *ida_mem, long int *nsteps)
    int IDAGetNumResEvals(void *ida_mem, long int *nrevals)
    int IDAGetNumLinSolvSetups(void *ida_mem, long int *nlinsetups)
    int IDAGetNumErrTestFails(void *ida_mem, long int *netfails)
    int IDAGetNumBacktrackOps(void *ida_mem, long int *nbacktr)
    int IDAGetConsistentIC(void *ida_mem, N_Vector yy0_mod, N_Vector yp0_mod)
    int IDAGetLastOrder(void *ida_mem, int *klast)
    int IDAGetCurrentOrder(void *ida_mem, int *kcur)
    int IDAGetCurrentCj(void *ida_mem, realtype *cj)
    int IDAGetCurrentY(void *ida_mem, N_Vector *ycur)
    int IDAGetCurrentYp(void *ida_mem, N_Vector *ypcur)
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
                              realtype *hinused, realtype *hlast,
                              realtype *hcur, realtype *tcur)
    int IDAGetNumNonlinSolvIters(void *ida_mem, long int *nniters)
    int IDAGetNumNonlinSolvConvFails(void *ida_mem, long int *nncfails)
    int IDAGetNonlinSolvStats(void *ida_mem, long int *nniters,
                              long int *nncfails)
    char *IDAGetReturnFlagName(long int flag)

    void IDAFree(void **ida_mem)

cdef extern from "ida/ida_ls.h":
    enum: IDALS_SUCCESS          # 0
    enum: IDALS_MEM_NULL         #-1
    enum: IDALS_LMEM_NULL        #-2
    enum: IDALS_ILL_INPUT        #-3
    enum: IDALS_MEM_FAIL         #-4
    enum: IDALS_PMEM_NULL        #-5
    enum: IDALS_JACFUNC_UNRECVR  #-6
    enum: IDALS_JACFUNC_RECVR    #-7
    enum: IDALS_SUNMAT_FAIL      #-8
    enum: IDALS_SUNLS_FAIL       #-9

    ctypedef int (*IDALsJacFn)(realtype t, realtype c_j, N_Vector y,
                               N_Vector yp, N_Vector r, SUNMatrix Jac,
                               void *user_data, N_Vector tmp1,
                               N_Vector tmp2, N_Vector tmp3)

    ctypedef int (*IDALsPrecSetupFn)(realtype tt, N_Vector yy,
                                    N_Vector yp, N_Vector rr,
                                    realtype c_j, void *user_data)

    ctypedef int (*IDALsPrecSolveFn)(realtype tt, N_Vector yy,
                                    N_Vector yp, N_Vector rr,
                                    N_Vector rvec, N_Vector zvec,
                                    realtype c_j, realtype delta,
                                    void *user_data)

    ctypedef int (*IDALsJacTimesSetupFn)(realtype tt, N_Vector yy,
                                        N_Vector yp, N_Vector rr,
                                        realtype c_j, void *user_data) except? -1

    ctypedef int (*IDALsJacTimesVecFn)(realtype tt, N_Vector yy,
                                      N_Vector yp, N_Vector rr,
                                      N_Vector v, N_Vector Jv,
                                      realtype c_j, void *user_data,
                                      N_Vector tmp1, N_Vector tmp2) except? -1

    int IDASetLinearSolver(void *ida_mem, SUNLinearSolver LS, SUNMatrix A)
    
    int IDASetJacFn(void *ida_mem, IDALsJacFn jac)
    int IDASetPreconditioner(void *ida_mem, IDALsPrecSetupFn pset,
                             IDALsPrecSolveFn psolve)
    int IDASetJacTimes(void *ida_mem, IDALsJacTimesSetupFn jtsetup,
                       IDALsJacTimesVecFn jtimes)
    int IDASetEpsLin(void *ida_mem, realtype eplifac)
    int IDASetIncrementFactor(void *ida_mem, realtype dqincfac)

    int IDAGetLinWorkSpace(void *ida_mem, long int *lenrwLS, long int *leniwLS)
    int IDAGetNumJacEvals(void *ida_mem, long int *njevals)
    int IDAGetNumPrecEvals(void *ida_mem, long int *npevals)
    int IDAGetNumPrecSolves(void *ida_mem, long int *npsolves)
    int IDAGetNumLinIters(void *ida_mem, long int *nliters)
    int IDAGetNumLinConvFails(void *ida_mem, long int *nlcfails)
    int IDAGetNumJTSetupEvals(void *ida_mem, long int *njtsetups)
    int IDAGetNumJtimesEvals(void *ida_mem, long int *njvevals)
    int IDAGetNumLinResEvals(void *ida_mem, long int *nrevalsLS)
    int IDAGetLastLinFlag(void *ida_mem, long int *flag)
    char *IDAGetLinReturnFlagName(long int flag)

cdef extern from "ida/ida_direct.h":
    
    ctypedef IDALsJacFn IDADlsJacFn

    int IDADlsSetLinearSolver(void *ida_mem, SUNLinearSolver LS, SUNMatrix A)
    int IDADlsSetJacFn(void *ida_mem, IDADlsJacFn jac)
    
    int IDADlsGetWorkSpace(void *ida_mem, long int *lenrwLS, long int *leniwLS)
    int IDADlsGetNumJacEvals(void *ida_mem, long int *njevals)
    int IDADlsGetNumResEvals(void *ida_mem, long int *nfevalsLS)
    int IDADlsGetLastFlag(void *ida_mem, long int *flag)
    char *IDADlsGetReturnFlagName(long int flag)

cdef extern from "ida/ida_spils.h":

    ctypedef IDALsPrecSetupFn IDASpilsPrecSetupFn;
    ctypedef IDALsPrecSolveFn IDASpilsPrecSolveFn;
    ctypedef IDALsJacTimesSetupFn IDASpilsJacTimesSetupFn;
    ctypedef IDALsJacTimesVecFn IDASpilsJacTimesVecFn;

    int IDASpilsSetLinearSolver(void *ida_mem, SUNLinearSolver LS)
    int IDASpilsSetPreconditioner(void *ida_mem,
                                  IDASpilsPrecSetupFn pset,
                                  IDASpilsPrecSolveFn psolve)
    int IDASpilsSetJacTimes(void *ida_mem, IDASpilsJacTimesSetupFn jtsetup,
                            IDASpilsJacTimesVecFn jtimes)
    int IDASpilsSetEpsLin(void *ida_mem, realtype eplifac)
    int IDASpilsSetIncrementFactor(void *ida_mem, realtype dqincfac)
    int IDASpilsGetWorkSpace(void *ida_mem, long int *lenrwLS, long int *leniwLS)
    int IDASpilsGetNumPrecEvals(void *ida_mem, long int *npevals)
    int IDASpilsGetNumPrecSolves(void *ida_mem, long int *npsolves)
    int IDASpilsGetNumLinIters(void *ida_mem, long int *nliters)
    int IDASpilsGetNumConvFails(void *ida_mem, long int *nlcfails)
    int IDASpilsGetNumJTSetupEvals(void *ida_mem, long int *njtsetups)
    int IDASpilsGetNumJtimesEvals(void *ida_mem, long int *njvevals)
    int IDASpilsGetNumResEvals(void *ida_mem, long int *nrevalsLS)
    int IDASpilsGetLastFlag(void *ida_mem, long int *flag)
    char *IDASpilsGetReturnFlagName(long int flag)

cdef extern from "ida/ida_bbdpre.h":

    ctypedef int (*IDABBDLocalFn)(sunindextype Nlocal, realtype tt,
                                  N_Vector yy, N_Vector yp, N_Vector gval,
                                  void *user_data)
    ctypedef int (*IDABBDCommFn)(sunindextype Nlocal, realtype tt,
                                 N_Vector yy, N_Vector yp,
                                 void *user_data)

    int IDABBDPrecInit(void *ida_mem, sunindextype Nlocal,
                       sunindextype mudq, sunindextype mldq,
                       sunindextype mukeep, sunindextype mlkeep,
                       realtype dq_rel_yy,
                       IDABBDLocalFn Gres, IDABBDCommFn Gcomm)
    int IDABBDPrecReInit(void *ida_mem,
                         sunindextype mudq, sunindextype mldq,
                         realtype dq_rel_yy)
    int IDABBDPrecGetWorkSpace(void *ida_mem,
                               long int *lenrwBBDP, long int *leniwBBDP)
    int IDABBDPrecGetNumGfnEvals(void *ida_mem, long int *ngevalsBBDP)
