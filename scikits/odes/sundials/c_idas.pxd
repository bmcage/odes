from .c_sundials cimport *
from libc.stdio cimport FILE

cdef extern from "idas/idas.h":
    enum: IDA_NORMAL          # 1
    enum: IDA_ONE_STEP        # 2

    #/* icopt */
    enum: IDA_YA_YDP_INIT     # 1
    enum: IDA_Y_INIT          # 2

    #/* ism */
    enum: IDA_SIMULTANEOUS    # 1
    enum: IDA_STAGGERED       # 2

    #/* DQtype */
    enum: IDA_CENTERED        # 1
    enum: IDA_FORWARD         # 2

    #/* interp */
    enum: IDA_HERMITE         # 1
    enum: IDA_POLYNOMIAL      # 2

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

    enum: IDA_NO_QUAD         # -30
    enum: IDA_QRHS_FAIL       # -31
    enum: IDA_FIRST_QRHS_ERR  # -32
    enum: IDA_REP_QRHS_ERR    # -33

    enum: IDA_NO_SENS         # -40
    enum: IDA_SRES_FAIL       # -41
    enum: IDA_REP_SRES_ERR    # -42
    enum: IDA_BAD_IS          # -43

    enum: IDA_NO_QUADSENS     # -50
    enum: IDA_QSRHS_FAIL      # -51
    enum: IDA_FIRST_QSRHS_ERR # -52
    enum: IDA_REP_QSRHS_ERR   # -53

    enum: IDA_UNRECOGNIZED_ERROR #-99

    enum: IDA_NO_ADJ          # -101
    enum: IDA_NO_FWD          # -102
    enum: IDA_NO_BCK          # -103
    enum: IDA_BAD_TB0         # -104
    enum: IDA_REIFWD_FAIL     # -105
    enum: IDA_FWD_FAIL        # -106
    enum: IDA_GETY_BADT       # -107


    ctypedef int (*IDAResFn)(sunrealtype tt, N_Vector yy, N_Vector yp,
                    N_Vector rr, void *user_data)

    ctypedef int (*IDARootFn)(sunrealtype t, N_Vector y, N_Vector yp,
                    sunrealtype *gout, void *user_data)

    ctypedef int (*IDAEwtFn)(N_Vector y, N_Vector ewt, void *user_data)

    ctypedef void (*IDAErrHandlerFn)(int error_code, const char *module, 
                                    const char *function, char *msg, 
                                    void *user_data)

    
    ctypedef int (*IDAQuadRhsFn)(sunrealtype tres, N_Vector yy, N_Vector yp,
                                 N_Vector rrQ, void *user_data)
    
    ctypedef int (*IDASensResFn)(int Ns, sunrealtype t,
                                 N_Vector yy, N_Vector yp, N_Vector resval,
                                 N_Vector *yyS, N_Vector *ypS,
                                 N_Vector *resvalS, void *user_data,
                                 N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
    
    ctypedef int (*IDAQuadSensRhsFn)(int Ns, sunrealtype t,
                                     N_Vector yy, N_Vector yp,
                                     N_Vector *yyS, N_Vector *ypS,
                                     N_Vector rrQ, N_Vector *rhsvalQS,
                                     void *user_data,
                                     N_Vector yytmp, N_Vector yptmp, 
                                     N_Vector tmpQS)
    
    ctypedef int (*IDAResFnB)(sunrealtype tt,
                              N_Vector yy, N_Vector yp,
                              N_Vector yyB, N_Vector ypB,
                              N_Vector rrB, void *user_dataB)
    
    ctypedef int (*IDAResFnBS)(sunrealtype t,
                               N_Vector yy, N_Vector yp,
                               N_Vector *yyS, N_Vector *ypS,
                               N_Vector yyB, N_Vector ypB,
                               N_Vector rrBS, void *user_dataB)
    
    ctypedef int (*IDAQuadRhsFnB)(sunrealtype tt,
                                  N_Vector yy, N_Vector yp,
                                  N_Vector yyB, N_Vector ypB,
                                  N_Vector rhsvalBQ, void *user_dataB)
    
    ctypedef int (*IDAQuadRhsFnBS)(sunrealtype t,
                                   N_Vector yy, N_Vector yp,
                                   N_Vector *yyS, N_Vector *ypS,
                                   N_Vector yyB, N_Vector ypB,
                                   N_Vector rhsvalBQS, void *user_dataB)

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


    #* Exported Functions -- Quadrature
    int IDAQuadInit(void *ida_mem, IDAQuadRhsFn rhsQ, N_Vector yQ0)
    int IDAQuadReInit(void *ida_mem, N_Vector yQ0)

    int IDAQuadSStolerances(void *ida_mem, sunrealtype reltolQ, sunrealtype abstolQ)
    int IDAQuadSVtolerances(void *ida_mem, sunrealtype reltolQ, N_Vector abstolQ)

    int IDASetQuadErrCon(void *ida_mem, sunbooleantype errconQ)

    int IDAGetQuad(void *ida_mem, sunrealtype *t, N_Vector yQout)
    int IDAGetQuadDky(void *ida_mem, sunrealtype t, int k, N_Vector dky)

    int IDAGetQuadNumRhsEvals(void *ida_mem, long int *nrhsQevals)
    int IDAGetQuadNumErrTestFails(void *ida_mem, long int *nQetfails)
    int IDAGetQuadErrWeights(void *ida_mem, N_Vector eQweight)
    int IDAGetQuadStats(void *ida_mem, long int *nrhsQevals, long int *nQetfails)

    void IDAQuadFree(void *ida_mem)

    # Exported Functions -- Sensitivities
    int IDASensInit(void *ida_mem, int Ns, int ism,
                    IDASensResFn resS, N_Vector *yS0, N_Vector *ypS0)
    int IDASensReInit(void *ida_mem, int ism, N_Vector *yS0, N_Vector *ypS0)

    int IDASensSStolerances(void *ida_mem, sunrealtype reltolS, sunrealtype *abstolS)
    int IDASensSVtolerances(void *ida_mem, sunrealtype reltolS, N_Vector *abstolS)
    int IDASensEEtolerances(void *ida_mem)

    int IDAGetSensConsistentIC(void *ida_mem, N_Vector *yyS0, N_Vector *ypS0)

    int IDASetSensDQMethod(void *ida_mem, int DQtype, sunrealtype DQrhomax)
    int IDASetSensErrCon(void *ida_mem, sunbooleantype errconS)
    int IDASetSensMaxNonlinIters(void *ida_mem, int maxcorS)
    int IDASetSensParams(void *ida_mem, sunrealtype *p, sunrealtype *pbar, int *plist)

    int IDASetNonlinearSolverSensSim(void *ida_mem, SUNNonlinearSolver NLS)
    int IDASetNonlinearSolverSensStg(void *ida_mem, SUNNonlinearSolver NLS)

    int IDASensToggleOff(void *ida_mem)

    int IDAGetSens(void *ida_mem, sunrealtype *tret, N_Vector *yySout)
    int IDAGetSens1(void *ida_mem, sunrealtype *tret, int iss, N_Vector yySret) # rename is to iss, is protected in python

    int IDAGetSensDky(void *ida_mem, sunrealtype t, int k, N_Vector *dkyS)
    int IDAGetSensDky1(void *ida_mem, sunrealtype t, int k, int iss, N_Vector dkyS) # rename is to iss, is protected in python

    int IDAGetSensNumResEvals(void *ida_mem, long int *nresSevals)
    int IDAGetNumResEvalsSens(void *ida_mem, long int *nresevalsS)
    int IDAGetSensNumErrTestFails(void *ida_mem, long int *nSetfails)
    int IDAGetSensNumLinSolvSetups(void *ida_mem, long int *nlinsetupsS)
    int IDAGetSensErrWeights(void *ida_mem, N_Vector_S eSweight)
    int IDAGetSensStats(void *ida_mem, long int *nresSevals,
                        long int *nresevalsS, long int *nSetfails,
                        long int *nlinsetupsS)
    int IDAGetSensNumNonlinSolvIters(void *ida_mem, long int *nSniters)
    int IDAGetSensNumNonlinSolvConvFails(void *ida_mem, long int *nSncfails)
    int IDAGetSensNonlinSolvStats(void *ida_mem, long int *nSniters,
                                  long int *nSncfails)

    void IDASensFree(void *ida_mem)


    # Exported Functions -- Sensitivity dependent quadrature
    int IDAQuadSensInit(void *ida_mem, IDAQuadSensRhsFn resQS,
                        N_Vector *yQS0)
    int IDAQuadSensReInit(void *ida_mem, N_Vector *yQS0)

    int IDAQuadSensSStolerances(void *ida_mem, sunrealtype reltolQS,
                                            sunrealtype *abstolQS)
    int IDAQuadSensSVtolerances(void *ida_mem, sunrealtype reltolQS,
                                N_Vector *abstolQS)
    int IDAQuadSensEEtolerances(void *ida_mem)

    int IDASetQuadSensErrCon(void *ida_mem, sunbooleantype errconQS)

    int IDAGetQuadSens(void *ida_mem, sunrealtype *tret, N_Vector *yyQSout)
    int IDAGetQuadSens1(void *ida_mem, sunrealtype *tret, int iss, N_Vector yyQSret) # rename is to iss, is protected in python
    int IDAGetQuadSensDky(void *ida_mem, sunrealtype t, int k, N_Vector *dkyQS)
    int IDAGetQuadSensDky1(void *ida_mem, sunrealtype t, int k, int iss,
                           N_Vector dkyQS) # rename is to iss, is protected in python

    int IDAGetQuadSensNumRhsEvals(void *ida_mem, long int *nrhsQSevals)
    int IDAGetQuadSensNumErrTestFails(void *ida_mem, long int *nQSetfails)
    int IDAGetQuadSensErrWeights(void *ida_mem, N_Vector *eQSweight)
    int IDAGetQuadSensStats(void *ida_mem,  long int *nrhsQSevals,
                            long int *nQSetfails)

    void IDAQuadSensFree(void* ida_mem)

    # Exported Functions -- Backward Problems
    int IDAAdjInit(void *ida_mem, long int steps, int interp)
    int IDAAdjReInit(void *ida_mem)
    void IDAAdjFree(void *ida_mem)

    int IDACreateB(void *ida_mem, int *which)
    int IDAInitB(void *ida_mem, int which, IDAResFnB resB,
                 sunrealtype tB0, N_Vector yyB0, N_Vector ypB0)
    int IDAInitBS(void *ida_mem, int which, IDAResFnBS resS,
                  sunrealtype tB0, N_Vector yyB0, N_Vector ypB0)
    int IDAReInitB(void *ida_mem, int which,
                            sunrealtype tB0, N_Vector yyB0, N_Vector ypB0)
    int IDASStolerancesB(void *ida_mem, int which,
                         sunrealtype relTolB, sunrealtype absTolB)
    int IDASVtolerancesB(void *ida_mem, int which,
                         sunrealtype relTolB, N_Vector absTolB)
    int IDAQuadInitB(void *ida_mem, int which,
                     IDAQuadRhsFnB rhsQB, N_Vector yQB0)
    int IDAQuadInitBS(void *ida_mem, int which,
                      IDAQuadRhsFnBS rhsQS, N_Vector yQB0)
    int IDAQuadReInitB(void *ida_mem, int which, N_Vector yQB0)
    int IDAQuadSStolerancesB(void *ida_mem, int which,
                             sunrealtype reltolQB, sunrealtype abstolQB)
    int IDAQuadSVtolerancesB(void *ida_mem, int which,
                             sunrealtype reltolQB, N_Vector abstolQB)

    int IDACalcICB (void *ida_mem, int which, sunrealtype tout1,
                    N_Vector yy0, N_Vector yp0)
    int IDACalcICBS(void *ida_mem, int which, sunrealtype tout1,
                    N_Vector yy0, N_Vector yp0,
                    N_Vector *yyS0, N_Vector *ypS0)

    int IDASolveF(void *ida_mem, sunrealtype tout, sunrealtype *tret,
                  N_Vector yret, N_Vector ypret, int itask, int *ncheckPtr)

    int IDASolveB(void *ida_mem, sunrealtype tBout, int itaskB)

    int IDAAdjSetNoSensi(void *ida_mem)

    int IDASetUserDataB(void *ida_mem, int which, void *user_dataB)
    int IDASetMaxOrdB(void *ida_mem, int which, int maxordB)
    int IDASetMaxNumStepsB(void *ida_mem, int which, long int mxstepsB)
    int IDASetInitStepB(void *ida_mem, int which, sunrealtype hinB)
    int IDASetMaxStepB(void *ida_mem, int which, sunrealtype hmaxB)
    int IDASetSuppressAlgB(void *ida_mem, int which, sunbooleantype suppressalgB)
    int IDASetIdB(void *ida_mem, int which, N_Vector idB)
    int IDASetConstraintsB(void *ida_mem, int which, N_Vector constraintsB)
    int IDASetQuadErrConB(void *ida_mem, int which, int errconQB)

    int IDASetNonlinearSolverB(void *ida_mem, int which, SUNNonlinearSolver NLS)

    int IDAGetB(void* ida_mem, int which, sunrealtype *tret,
                N_Vector yy, N_Vector yp)
    int IDAGetQuadB(void *ida_mem, int which, sunrealtype *tret, N_Vector qB)

    void *IDAGetAdjIDABmem(void *ida_mem, int which)

    int IDAGetConsistentICB(void *ida_mem, int which, 
                            N_Vector yyB0, N_Vector ypB0)

    int IDAGetAdjY(void *ida_mem, sunrealtype t, N_Vector yy, N_Vector yp)

    struct _IDAadjCheckPointRec:
        void *my_addr
        void *next_addr
        sunrealtype t0
        sunrealtype t1
        long int nstep
        int order
        sunrealtype step

    ctypedef _IDAadjCheckPointRec IDAadjCheckPointRec
    
    int IDAGetAdjCheckPointsInfo(void *ida_mem, IDAadjCheckPointRec *ckpnt)

    int IDAGetAdjDataPointHermite(void *ida_mem, int which,
                                  sunrealtype *t, N_Vector yy, N_Vector yd)

    int IDAGetAdjDataPointPolynomial(void *ida_mem, int which,
                                     sunrealtype *t, int *order, N_Vector y)

    int IDAGetAdjCurrentCheckPoint(void *ida_mem, void **addr)


cdef extern from "idas/idas_ls.h":
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

    enum: IDALS_NO_ADJ          # -101
    enum: IDALS_LMEMB_NULL      # -102

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

    ctypedef int (*IDALsJacFnB)(sunrealtype tt, sunrealtype c_jB, N_Vector yy,
                                N_Vector yp, N_Vector yyB, N_Vector ypB,
                                N_Vector rrB, SUNMatrix JacB,
                                void *user_dataB, N_Vector tmp1B,
                                N_Vector tmp2B, N_Vector tmp3B)
    
    ctypedef int (*IDALsJacFnBS)(sunrealtype tt, sunrealtype c_jB, N_Vector yy,
                                 N_Vector yp, N_Vector *yS, N_Vector *ypS,
                                 N_Vector yyB, N_Vector ypB, N_Vector rrB,
                                 SUNMatrix JacB, void *user_dataB,
                                 N_Vector tmp1B, N_Vector tmp2B,
                                 N_Vector tmp3B)
    
    ctypedef int (*IDALsPrecSetupFnB)(sunrealtype tt, N_Vector yy,
                                      N_Vector yp, N_Vector yyB,
                                      N_Vector ypB, N_Vector rrB,
                                      sunrealtype c_jB, void *user_dataB)
    
    ctypedef int (*IDALsPrecSetupFnBS)(sunrealtype tt, N_Vector yy,
                                       N_Vector yp, N_Vector *yyS,
                                       N_Vector *ypS, N_Vector yyB,
                                       N_Vector ypB, N_Vector rrB,
                                       sunrealtype c_jB, void *user_dataB)
    
    ctypedef int (*IDALsPrecSolveFnB)(sunrealtype tt, N_Vector yy,
                                      N_Vector yp, N_Vector yyB,
                                      N_Vector ypB, N_Vector rrB,
                                      N_Vector rvecB, N_Vector zvecB,
                                      sunrealtype c_jB, sunrealtype deltaB,
                                      void *user_dataB)
    
    ctypedef int (*IDALsPrecSolveFnBS)(sunrealtype tt, N_Vector yy,
                                       N_Vector yp, N_Vector *yyS,
                                       N_Vector *ypS, N_Vector yyB,
                                       N_Vector ypB, N_Vector rrB,
                                       N_Vector rvecB, N_Vector zvecB,
                                       sunrealtype c_jB, sunrealtype deltaB,
                                       void *user_dataB)
    
    ctypedef int (*IDALsJacTimesSetupFnB)(sunrealtype t, N_Vector yy,
                                          N_Vector yp, N_Vector yyB,
                                          N_Vector ypB, N_Vector rrB,
                                          sunrealtype c_jB, void *user_dataB)
    
    ctypedef int (*IDALsJacTimesSetupFnBS)(sunrealtype t, N_Vector yy,
                                           N_Vector yp, N_Vector *yyS,
                                           N_Vector *ypS, N_Vector yyB,
                                           N_Vector ypB, N_Vector rrB,
                                           sunrealtype c_jB, void *user_dataB)
    
    ctypedef int (*IDALsJacTimesVecFnB)(sunrealtype t, N_Vector yy,
                                        N_Vector yp, N_Vector yyB,
                                        N_Vector ypB, N_Vector rrB,
                                        N_Vector vB, N_Vector JvB,
                                        sunrealtype c_jB, void *user_dataB,
                                        N_Vector tmp1B, N_Vector tmp2B)
    
    ctypedef int (*IDALsJacTimesVecFnBS)(sunrealtype t, N_Vector yy,
                                         N_Vector yp, N_Vector *yyS,
                                         N_Vector *ypS, N_Vector yyB,
                                         N_Vector ypB, N_Vector rrB,
                                         N_Vector vB, N_Vector JvB,
                                         sunrealtype c_jB, void *user_dataB,
                                         N_Vector tmp1B, N_Vector tmp2B)

    int IDASetLinearSolverB(void *ida_mem, int which, SUNLinearSolver LS,
                            SUNMatrix A)

    int IDASetJacFnB(void *ida_mem, int which, IDALsJacFnB jacB)
    int IDASetJacFnBS(void *ida_mem, int which, IDALsJacFnBS jacBS)

    int IDASetEpsLinB(void *ida_mem, int which, sunrealtype eplifacB)
    int IDASetIncrementFactorB(void *ida_mem, int which, sunrealtype dqincfacB)
    int IDASetPreconditionerB(void *ida_mem, int which,
                              IDALsPrecSetupFnB psetB,
                              IDALsPrecSolveFnB psolveB)
    int IDASetPreconditionerBS(void *ida_mem, int which,
                               IDALsPrecSetupFnBS psetBS,
                               IDALsPrecSolveFnBS psolveBS)
    int IDASetJacTimesB(void *ida_mem, int which,
                        IDALsJacTimesSetupFnB jtsetupB,
                        IDALsJacTimesVecFnB jtimesB)
    int IDASetJacTimesBS(void *ida_mem, int which,
                         IDALsJacTimesSetupFnBS jtsetupBS,
                         IDALsJacTimesVecFnBS jtimesBS)


cdef extern from "idas/idas_direct.h":
    
    ctypedef IDALsJacFn IDAJacFn
    ctypedef IDALsJacFnB IDAJacFnB
    ctypedef IDALsJacFnBS IDAJacFnBS

    int IDASetJacFn(void *ida_mem, IDAJacFn jac)
    
    int IDAGetWorkSpace(void *ida_mem, long int *lenrwLS, long int *leniwLS)
    int IDAGetNumJacEvals(void *ida_mem, long int *njevals)
    int IDAGetNumResEvals(void *ida_mem, long int *nfevalsLS)
    int IDAGetLastFlag(void *ida_mem, long int *flag)
    char *IDAGetReturnFlagName(long int flag)

    int IDASetJacFnB(void *ida_mem, int which, IDAJacFnB jacB)
    
    int IDASetJacFnBS(void *ida_mem, int which, IDAJacFnBS jacBS)


cdef extern from "idas/idas_spils.h":

    ctypedef IDALsPrecSetupFn IDAPrecSetupFn;
    ctypedef IDALsPrecSolveFn IDAPrecSolveFn;
    ctypedef IDALsJacTimesSetupFn IDAJacTimesSetupFn;
    ctypedef IDALsJacTimesVecFn IDAJacTimesVecFn;

    ctypedef IDALsPrecSetupFnB IDAPrecSetupFnB
    ctypedef IDALsPrecSetupFnBS IDAPrecSetupFnBS
    ctypedef IDALsPrecSolveFnB IDAPrecSolveFnB
    ctypedef IDALsPrecSolveFnBS IDAPrecSolveFnBS
    ctypedef IDALsJacTimesSetupFnB IDAJacTimesSetupFnB
    ctypedef IDALsJacTimesSetupFnBS IDAJacTimesSetupFnBS
    ctypedef IDALsJacTimesVecFnB IDAJacTimesVecFnB
    ctypedef IDALsJacTimesVecFnBS IDAJacTimesVecFnBS

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

    int IDASetEpsLinB(void *ida_mem, int which, sunrealtype eplifacB)
    
    int IDASetIncrementFactorB(void *ida_mem, int which,
                                    sunrealtype dqincfacB)
    
    int IDASetPreconditionerB(void *ida_mem, int which,
                                   IDAPrecSetupFnB psetB,
                                   IDAPrecSolveFnB psolveB)
    
    int IDASetPreconditionerBS(void *ida_mem, int which,
                                    IDAPrecSetupFnBS psetBS,
                                    IDAPrecSolveFnBS psolveBS)
    
    int IDASetJacTimesB(void *ida_mem, int which,
                             IDAJacTimesSetupFnB jtsetupB,
                             IDAJacTimesVecFnB jtimesB)
    
    int IDASetJacTimesBS(void *ida_mem, int which,
                              IDAJacTimesSetupFnBS jtsetupBS,
                              IDAJacTimesVecFnBS jtimesBS)

cdef extern from "idas/idas_bbdpre.h":

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

    ctypedef int (*IDABBDLocalFnB)(sunindextype NlocalB, sunrealtype tt,
                                   N_Vector yy, N_Vector yp,
                                   N_Vector yyB, N_Vector ypB,
                                   N_Vector gvalB, void *user_dataB)
    
    ctypedef int (*IDABBDCommFnB)(sunindextype NlocalB, sunrealtype tt,
                                  N_Vector yy, N_Vector yp,
                                  N_Vector yyB, N_Vector ypB, void *user_dataB)

    int IDABBDPrecInitB(void *ida_mem, int which, sunindextype NlocalB,
                        sunindextype mudqB, sunindextype mldqB,
                        sunindextype mukeepB, sunindextype mlkeepB,
                        sunrealtype dq_rel_yyB,
                        IDABBDLocalFnB GresB, IDABBDCommFnB GcommB)

    int IDABBDPrecReInitB(void *ida_mem, int which,
                          sunindextype mudqB, sunindextype mldqB,
                          sunrealtype dq_rel_yyB)
