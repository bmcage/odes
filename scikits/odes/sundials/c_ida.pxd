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

    enum: IDA_MEM_NULL        #-20
    enum: IDA_MEM_FAIL        #-21
    enum: IDA_ILL_INPUT       #-22
    enum: IDA_NO_MALLOC       #-23
    enum: IDA_BAD_EWT         #-24
    enum: IDA_BAD_K           #-25
    enum: IDA_BAD_T           #-26
    enum: IDA_BAD_DKY         #-27

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

cdef extern from "ida/ida_direct.h":
    enum: IDADLS_SUCCESS
    enum: IDADLS_MEM_NULL
    enum: IDADLS_LMEM_NULL
    enum: IDADLS_ILL_INPUT
    enum: IDADLS_MEM_FAIL

    #/* Additional last_flag values */
    enum: IDADLS_JACFUNC_UNRECVR
    enum: IDADLS_JACFUNC_RECVR

    ctypedef int (*IDADlsDenseJacFn)(long int N, realtype t, realtype c_j,
                                     N_Vector y, N_Vector yp, N_Vector r,
                                     DlsMat Jac, void *user_data, N_Vector tmp1,
                                     N_Vector tmp2, N_Vector tmp3)

    ctypedef int (*IDADlsBandJacFn)(long int N, long int mupper, long int mlower,
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
    int IDADlsGetLastFlag(void *ida_mem, long int *flag)
    char *IDADlsGetReturnFlagName(long int flag)

cdef extern from "ida/ida_band.h":
    int IDABand(void *ida_mem, long int Neq, long int mupper, long int mlower)

cdef extern from "ida/ida_dense.h":
     int IDADense(void *ida_mem, long int Neq)

cdef extern from "ida/ida_lapack.h":
     int IDALapackDense(void *ida_mem, int N)
     int IDALapackBand(void *ida_mem, int N, int mupper, int mlower)

cdef extern from "ida/ida_spils.h":


    enum: IDASPILS_SUCCESS   #  0
    enum: IDASPILS_MEM_NULL  # -1
    enum: IDASPILS_LMEM_NULL # -2
    enum: IDASPILS_ILL_INPUT # -3
    enum: IDASPILS_MEM_FAIL  # -4
    enum: IDASPILS_PMEM_NULL # -5

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
    int IDASpilsGetLastFlag(void *ida_mem, long int *flag)
    char *IDASpilsGetReturnFlagName(long int flag)

cdef extern from "ida/ida_spgmr.h":
     int IDASpgmr(void *ida_mem, int maxl)

cdef extern from "ida/ida_spbcgs.h":
     int IDASpbcg(void *ida_mem, int maxl)

cdef extern from "ida/ida_sptfqmr.h":
     int IDASptfqmr(void *ida_mem, int maxl)

cdef extern from "ida/ida_bbdpre.h":
    ctypedef int (*IDABBDLocalFn)(long int Nlocal, realtype tt,
                                  N_Vector yy, N_Vector yp, N_Vector gval,
                                  void *user_data)
    ctypedef int (*IDABBDCommFn)(long int Nlocal, realtype tt,
                                 N_Vector yy, N_Vector yp,
                                 void *user_data)

    int IDABBDPrecInit(void *ida_mem, long int Nlocal,
                       long int mudq, long int mldq,
                       long int mukeep, long int mlkeep,
                       realtype dq_rel_yy,
                       IDABBDLocalFn Gres, IDABBDCommFn Gcomm)
    int IDABBDPrecReInit(void *ida_mem,
                         long int mudq, long int mldq,
                         realtype dq_rel_yy)
    int IDABBDPrecGetWorkSpace(void *ida_mem,
                               long int *lenrwBBDP, long int *leniwBBDP)
    int IDABBDPrecGetNumGfnEvals(void *ida_mem, long int *ngevalsBBDP)
