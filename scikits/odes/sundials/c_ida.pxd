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

    ctypedef int (*IDAResFn)(sunrealtype tt, N_Vector yy, N_Vector yp,
                    N_Vector rr, void *user_data)

    ctypedef int (*IDARootFn)(sunrealtype t, N_Vector y, N_Vector yp,
                    sunrealtype *gout, void *user_data)

    ctypedef int (*IDAEwtFn)(N_Vector y, N_Vector ewt, void *user_data)

    ctypedef void (*IDAErrHandlerFn)(int error_code, const char *module, 
                                    const char *function, char *msg, 
                                    void *user_data)

    void *IDACreate(SUNContext sunctx)

    int IDAInit(void *ida_mem, IDAResFn res,
                sunrealtype t0, N_Vector yy0, N_Vector yp0)
    int IDAReInit(void *ida_mem,
                  sunrealtype t0, N_Vector yy0, N_Vector yp0)
    int IDASStolerances(void *ida_mem, sunrealtype reltol, sunrealtype abstol)
    int IDASVtolerances(void *ida_mem, sunrealtype reltol, N_Vector abstol)
    int IDAWFtolerances(void *ida_mem, IDAEwtFn efun)
    int IDACalcIC(void *ida_mem, int icopt, sunrealtype tout1)
    
    int IDASetNonlinConvCoefIC(void *ida_mem, sunrealtype epiccon)
    int IDASetMaxNumStepsIC(void *ida_mem, int maxnh)
    int IDASetMaxNumJacsIC(void *ida_mem, int maxnj)
    int IDASetMaxNumItersIC(void *ida_mem, int maxnit)
    int IDASetLineSearchOffIC(void *ida_mem, sunbooleantype lsoff)
    int IDASetStepToleranceIC(void *ida_mem, sunrealtype steptol)
    int IDASetMaxBacksIC(void *ida_mem, int maxbacks)
    
    int IDASetErrHandlerFn(void *ida_mem, IDAErrHandlerFn ehfun, void *eh_data)
    int IDASetErrFile(void *ida_mem, FILE *errfp)
    int IDASetUserData(void *ida_mem, void *user_data)
    int IDASetMaxOrd(void *ida_mem, int maxord)
    int IDASetMaxNumSteps(void *ida_mem, long int mxsteps)
    int IDASetInitStep(void *ida_mem, sunrealtype hin)
    int IDASetMaxStep(void *ida_mem, sunrealtype hmax)
    int IDASetStopTime(void *ida_mem, sunrealtype tstop)
    int IDASetNonlinConvCoef(void *ida_mem, sunrealtype epcon)
    int IDASetMaxErrTestFails(void *ida_mem, int maxnef)
    int IDASetMaxNonlinIters(void *ida_mem, int maxcor)
    int IDASetMaxConvFails(void *ida_mem, int maxncf)
    int IDASetSuppressAlg(void *ida_mem, sunbooleantype suppressalg)
    int IDASetId(void *ida_mem, N_Vector id)
    int IDASetConstraints(void *ida_mem, N_Vector constraints)

    int IDASetNonlinearSolver(void *ida_mem, SUNNonlinearSolver NLS)

    int IDARootInit(void *ida_mem, int nrtfn, IDARootFn g)
    int IDASetRootDirection(void *ida_mem, int *rootdir)
    int IDASetNoInactiveRootWarn(void *ida_mem)

    int IDASolve(void *ida_mem, sunrealtype tout, sunrealtype *tret,
                 N_Vector yret, N_Vector ypret, int itask)
    
    int IDAComputeY(void *ida_mem, N_Vector ycor, N_Vector y)
    int IDAComputeYp(void *ida_mem, N_Vector ycor, N_Vector yp)
    
    int IDAGetDky(void *ida_mem, sunrealtype t, int k, N_Vector dky)

    int IDAGetWorkSpace(void *ida_mem, long int *lenrw, long int *leniw)
    int IDAGetNumSteps(void *ida_mem, long int *nsteps)
    int IDAGetNumResEvals(void *ida_mem, long int *nrevals)
    int IDAGetNumLinSolvSetups(void *ida_mem, long int *nlinsetups)
    int IDAGetNumErrTestFails(void *ida_mem, long int *netfails)
    int IDAGetNumBacktrackOps(void *ida_mem, long int *nbacktr)
    int IDAGetConsistentIC(void *ida_mem, N_Vector yy0_mod, N_Vector yp0_mod)
    int IDAGetLastOrder(void *ida_mem, int *klast)
    int IDAGetCurrentOrder(void *ida_mem, int *kcur)
    int IDAGetCurrentCj(void *ida_mem, sunrealtype *cj)
    int IDAGetCurrentY(void *ida_mem, N_Vector *ycur)
    int IDAGetCurrentYp(void *ida_mem, N_Vector *ypcur)
    int IDAGetActualInitStep(void *ida_mem, sunrealtype *hinused)
    int IDAGetLastStep(void *ida_mem, sunrealtype *hlast)
    int IDAGetCurrentStep(void *ida_mem, sunrealtype *hcur)
    int IDAGetCurrentTime(void *ida_mem, sunrealtype *tcur)
    int IDAGetTolScaleFactor(void *ida_mem, sunrealtype *tolsfact)
    int IDAGetErrWeights(void *ida_mem, N_Vector eweight)
    int IDAGetEstLocalErrors(void *ida_mem, N_Vector ele)
    int IDAGetNumGEvals(void *ida_mem, long int *ngevals)
    int IDAGetRootInfo(void *ida_mem, int *rootsfound)
    int IDAGetIntegratorStats(void *ida_mem, long int *nsteps,
                              long int *nrevals, long int *nlinsetups,
                              long int *netfails, int *qlast, int *qcur,
                              sunrealtype *hinused, sunrealtype *hlast,
                              sunrealtype *hcur, sunrealtype *tcur)
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

    ctypedef int (*IDALsJacFn)(sunrealtype t, sunrealtype c_j, N_Vector y,
                               N_Vector yp, N_Vector r, SUNMatrix Jac,
                               void *user_data, N_Vector tmp1,
                               N_Vector tmp2, N_Vector tmp3)

    ctypedef int (*IDALsPrecSetupFn)(sunrealtype tt, N_Vector yy,
                                    N_Vector yp, N_Vector rr,
                                    sunrealtype c_j, void *user_data)

    ctypedef int (*IDALsPrecSolveFn)(sunrealtype tt, N_Vector yy,
                                    N_Vector yp, N_Vector rr,
                                    N_Vector rvec, N_Vector zvec,
                                    sunrealtype c_j, sunrealtype delta,
                                    void *user_data)

    ctypedef int (*IDALsJacTimesSetupFn)(sunrealtype tt, N_Vector yy,
                                        N_Vector yp, N_Vector rr,
                                        sunrealtype c_j, void *user_data) except? -1

    ctypedef int (*IDALsJacTimesVecFn)(sunrealtype tt, N_Vector yy,
                                      N_Vector yp, N_Vector rr,
                                      N_Vector v, N_Vector Jv,
                                      sunrealtype c_j, void *user_data,
                                      N_Vector tmp1, N_Vector tmp2) except? -1

    int IDASetLinearSolver(void *ida_mem, SUNLinearSolver LS, SUNMatrix A)
    
    int IDASetJacFn(void *ida_mem, IDALsJacFn jac)
    int IDASetPreconditioner(void *ida_mem, IDALsPrecSetupFn pset,
                             IDALsPrecSolveFn psolve)
    int IDASetJacTimes(void *ida_mem, IDALsJacTimesSetupFn jtsetup,
                       IDALsJacTimesVecFn jtimes)
    int IDASetEpsLin(void *ida_mem, sunrealtype eplifac)
    int IDASetIncrementFactor(void *ida_mem, sunrealtype dqincfac)

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
    
    ctypedef IDALsJacFn IDAJacFn

    int IDASetJacFn(void *ida_mem, IDAJacFn jac)
    
    int IDAGetWorkSpace(void *ida_mem, long int *lenrwLS, long int *leniwLS)
    int IDAGetNumJacEvals(void *ida_mem, long int *njevals)
    int IDAGetNumResEvals(void *ida_mem, long int *nfevalsLS)
    int IDAGetLastFlag(void *ida_mem, long int *flag)
    char *IDAGetReturnFlagName(long int flag)

cdef extern from "ida/ida_spils.h":

    ctypedef IDALsPrecSetupFn IDAPrecSetupFn;
    ctypedef IDALsPrecSolveFn IDAPrecSolveFn;
    ctypedef IDALsJacTimesSetupFn IDAJacTimesSetupFn;
    ctypedef IDALsJacTimesVecFn IDAJacTimesVecFn;

    int IDASetPreconditioner(void *ida_mem,
                                  IDAPrecSetupFn pset,
                                  IDAPrecSolveFn psolve)
    int IDASetJacTimes(void *ida_mem, IDAJacTimesSetupFn jtsetup,
                            IDAJacTimesVecFn jtimes)
    int IDASetEpsLin(void *ida_mem, sunrealtype eplifac)
    int IDASetIncrementFactor(void *ida_mem, sunrealtype dqincfac)
    int IDAGetWorkSpace(void *ida_mem, long int *lenrwLS, long int *leniwLS)
    int IDAGetNumPrecEvals(void *ida_mem, long int *npevals)
    int IDAGetNumPrecSolves(void *ida_mem, long int *npsolves)
    int IDAGetNumLinIters(void *ida_mem, long int *nliters)
    int IDAGetNumConvFails(void *ida_mem, long int *nlcfails)
    int IDAGetNumJTSetupEvals(void *ida_mem, long int *njtsetups)
    int IDAGetNumJtimesEvals(void *ida_mem, long int *njvevals)
    int IDAGetNumResEvals(void *ida_mem, long int *nrevalsLS)
    int IDAGetLastFlag(void *ida_mem, long int *flag)
    char *IDAGetReturnFlagName(long int flag)

cdef extern from "ida/ida_bbdpre.h":

    ctypedef int (*IDABBDLocalFn)(sunindextype Nlocal, sunrealtype tt,
                                  N_Vector yy, N_Vector yp, N_Vector gval,
                                  void *user_data)
    ctypedef int (*IDABBDCommFn)(sunindextype Nlocal, sunrealtype tt,
                                 N_Vector yy, N_Vector yp,
                                 void *user_data)

    int IDABBDPrecInit(void *ida_mem, sunindextype Nlocal,
                       sunindextype mudq, sunindextype mldq,
                       sunindextype mukeep, sunindextype mlkeep,
                       sunrealtype dq_rel_yy,
                       IDABBDLocalFn Gres, IDABBDCommFn Gcomm)
    int IDABBDPrecReInit(void *ida_mem,
                         sunindextype mudq, sunindextype mldq,
                         sunrealtype dq_rel_yy)
    int IDABBDPrecGetWorkSpace(void *ida_mem,
                               long int *lenrwBBDP, long int *leniwBBDP)
    int IDABBDPrecGetNumGfnEvals(void *ida_mem, long int *ngevalsBBDP)
