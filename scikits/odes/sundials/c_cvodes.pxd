from .c_sundials cimport *
from libc.stdio cimport FILE

cdef extern from "cvodes/cvodes.h":
    # lmm
    enum: CV_ADAMS # 1
    enum: CV_BDF   # 2

    # itask
    enum: CV_NORMAL     # 1
    enum: CV_ONE_STEP   # 2

    # ism
    enum: CV_SIMULTANEOUS   # 1
    enum: CV_STAGGERED      # 2
    enum: CV_STAGGERED1     # 3

    # DQtype
    enum: CV_CENTERED       # 1
    enum: CV_FORWARD        # 2

    # interp
    enum: CV_HERMITE        # 1
    enum: CV_POLYNOMIAL     # 2

    # CVODE return flags
    enum: CV_SUCCESS           #    0
    enum: CV_TSTOP_RETURN      #    1
    enum: CV_ROOT_RETURN       #    2

    enum: CV_WARNING           #   99

    enum: CV_TOO_MUCH_WORK     #   -1
    enum: CV_TOO_MUCH_ACC      #   -2
    enum: CV_ERR_FAILURE       #   -3
    enum: CV_CONV_FAILURE      #   -4

    enum: CV_LINIT_FAIL        #   -5
    enum: CV_LSETUP_FAIL       #   -6
    enum: CV_LSOLVE_FAIL       #   -7
    enum: CV_RHSFUNC_FAIL      #   -8
    enum: CV_FIRST_RHSFUNC_ERR #   -9
    enum: CV_REPTD_RHSFUNC_ERR #   -10
    enum: CV_UNREC_RHSFUNC_ERR #   -11
    enum: CV_RTFUNC_FAIL       #   -12
    enum: CV_NLS_INIT_FAIL     #   -13
    enum: CV_NLS_SETUP_FAIL    #   -14
    enum: CV_CONSTR_FAIL       #   -15
    enum: CV_NLS_FAIL          #   -16

    enum: CV_MEM_FAIL          #   -20
    enum: CV_MEM_NULL          #   -21
    enum: CV_ILL_INPUT         #   -22
    enum: CV_NO_MALLOC         #   -23
    enum: CV_BAD_K             #   -24
    enum: CV_BAD_T             #   -25
    enum: CV_BAD_DKY           #   -26
    enum: CV_TOO_CLOSE         #   -27
    enum: CV_VECTOROP_ERR      #   -28
    
    enum: CV_NO_QUAD             # -30
    enum: CV_QRHSFUNC_FAIL       # -31
    enum: CV_FIRST_QRHSFUNC_ERR  # -32
    enum: CV_REPTD_QRHSFUNC_ERR  # -33
    enum: CV_UNREC_QRHSFUNC_ERR  # -34

    enum: CV_NO_SENS             # -40
    enum: CV_SRHSFUNC_FAIL       # -41
    enum: CV_FIRST_SRHSFUNC_ERR  # -42
    enum: CV_REPTD_SRHSFUNC_ERR  # -43
    enum: CV_UNREC_SRHSFUNC_ERR  # -44

    enum: CV_BAD_IS              # -45

    enum: CV_NO_QUADSENS         # -50
    enum: CV_QSRHSFUNC_FAIL      # -51
    enum: CV_FIRST_QSRHSFUNC_ERR # -52
    enum: CV_REPTD_QSRHSFUNC_ERR # -53
    enum: CV_UNREC_QSRHSFUNC_ERR # -54
    
    enum: CV_UNRECOGNIZED_ERR    # -99
    
    # adjoint return values 

    enum: CV_NO_ADJ             # -101
    enum: CV_NO_FWD             # -102
    enum: CV_NO_BCK             # -103
    enum: CV_BAD_TB0            # -104
    enum: CV_REIFWD_FAIL        # -105
    enum: CV_FWD_FAIL           # -106
    enum: CV_GETY_BADT          # -107

    ctypedef int (*CVRhsFn)(realtype t, N_Vector y, N_Vector ydot, void *user_data) except? -1
    ctypedef int (*CVRootFn)(realtype t, N_Vector y, realtype *gout, void *user_data) except? -1
    ctypedef int (*CVEwtFn)(N_Vector y, N_Vector ewt, void *user_data)
    ctypedef void (*CVErrHandlerFn)(int error_code,
                               char *module, char *function,
                               char *msg, void *user_data)

    ctypedef int (*CVQuadRhsFn)(realtype t, N_Vector y,
                                N_Vector yQdot, void *user_data)
    ctypedef int (*CVSensRhsFn)(int Ns, realtype t,  N_Vector y, N_Vector ydot,
                                N_Vector *yS, N_Vector *ySdot,
                                void *user_data,
                                N_Vector tmp1, N_Vector tmp2)
    ctypedef int (*CVSensRhs1Fn)(int Ns, realtype t, N_Vector y, N_Vector ydot,
                                 int iS, N_Vector yS, N_Vector ySdot,
                                 void *user_data,
                                 N_Vector tmp1, N_Vector tmp2)
    ctypedef int (*CVQuadSensRhsFn)(int Ns, realtype t, N_Vector y, 
                                    N_Vector *yS, N_Vector yQdot, 
                                    N_Vector *yQSdot, void *user_data,
                                    N_Vector tmp, N_Vector tmpQ)
    ctypedef int (*CVRhsFnB)(realtype t, N_Vector y, N_Vector yB,
                             N_Vector yBdot, void *user_dataB)
    ctypedef int (*CVRhsFnBS)(realtype t, N_Vector y, N_Vector *yS,
                              N_Vector yB, N_Vector yBdot, void *user_dataB)

    ctypedef int (*CVQuadRhsFnB)(realtype t, N_Vector y, N_Vector yB,
                                 N_Vector qBdot, void *user_dataB)
    ctypedef int (*CVQuadRhsFnBS)(realtype t, N_Vector y, N_Vector *yS,
                                 N_Vector yB, N_Vector qBdot, void *user_dataB)

    # Exported Functions -- Forward Problems
    void *CVodeCreate(int lmm)

    int CVodeInit(void *cvode_mem, CVRhsFn f, realtype t0, N_Vector y0)
    int CVodeReInit(void *cvode_mem, realtype t0, N_Vector y0)

    int CVodeSStolerances(void *cvode_mem, realtype reltol, realtype abstol)
    int CVodeSVtolerances(void *cvode_mem, realtype reltol, N_Vector abstol)
    int CVodeWFtolerances(void *cvode_mem, CVEwtFn efun)

    int CVodeSetErrHandlerFn(void *cvode_mem, CVErrHandlerFn ehfun, void *eh_data)
    int CVodeSetErrFile(void *cvode_mem, FILE *errfp)
    int CVodeSetUserData(void *cvode_mem, void *user_data)
    int CVodeSetMaxOrd(void *cvode_mem, int maxord)
    int CVodeSetMaxNumSteps(void *cvode_mem, long int mxsteps)
    int CVodeSetMaxHnilWarns(void *cvode_mem, int mxhnil)
    int CVodeSetStabLimDet(void *cvode_mem, booleantype stldet)
    int CVodeSetInitStep(void *cvode_mem, realtype hin)
    int CVodeSetMinStep(void *cvode_mem, realtype hmin)
    int CVodeSetMaxStep(void *cvode_mem, realtype hmax)
    int CVodeSetStopTime(void *cvode_mem, realtype tstop)
    int CVodeSetMaxErrTestFails(void *cvode_mem, int maxnef)
    int CVodeSetMaxNonlinIters(void *cvode_mem, int maxcor)
    int CVodeSetMaxConvFails(void *cvode_mem, int maxncf)
    int CVodeSetNonlinConvCoef(void *cvode_mem, realtype nlscoef)
    int CVodeSetConstraints(void *cvode_mem, N_Vector constraints)

    int CVodeSetNonlinearSolver(void *cvode_mem, SUNNonlinearSolver NLS)

    int CVodeRootInit(void *cvode_mem, int nrtfn, CVRootFn g)
    int CVodeSetRootDirection(void *cvode_mem, int *rootdir)
    int CVodeSetNoInactiveRootWarn(void *cvode_mem)

    int CVode(void *cvode_mem, realtype tout, N_Vector yout,
                          realtype *tret, int itask)
    int CVodeGetDky(void *cvode_mem, realtype t, int k, N_Vector dky)

    int CVodeGetWorkSpace(void *cvode_mem, long int *lenrw, long int *leniw)
    int CVodeGetNumSteps(void *cvode_mem, long int *nsteps)
    int CVodeGetNumRhsEvals(void *cvode_mem, long int *nfevals)
    int CVodeGetNumLinSolvSetups(void *cvode_mem, long int *nlinsetups)
    int CVodeGetNumErrTestFails(void *cvode_mem, long int *netfails)
    int CVodeGetLastOrder(void *cvode_mem, int *qlast)
    int CVodeGetCurrentOrder(void *cvode_mem, int *qcur)
    int CVodeGetCurrentGamma(void *cvode_mem, realtype *gamma)
    int CVodeGetNumStabLimOrderReds(void *cvode_mem, long int *nslred)
    int CVodeGetActualInitStep(void *cvode_mem, realtype *hinused)
    int CVodeGetLastStep(void *cvode_mem, realtype *hlast)
    int CVodeGetCurrentStep(void *cvode_mem, realtype *hcur)
    int CVodeGetCurrentState(void *cvode_mem, N_Vector *y)
    int CVodeGetCurrentTime(void *cvode_mem, realtype *tcur)
    int CVodeGetTolScaleFactor(void *cvode_mem, realtype *tolsfac)
    int CVodeGetErrWeights(void *cvode_mem, N_Vector eweight)
    int CVodeGetEstLocalErrors(void *cvode_mem, N_Vector ele)
    int CVodeGetNumGEvals(void *cvode_mem, long int *ngevals)
    int CVodeGetRootInfo(void *cvode_mem, int *rootsfound)

    int CVodeGetIntegratorStats(void *cvode_mem, long int *nsteps,
                                            long int *nfevals, long int *nlinsetups,
                                            long int *netfails, int *qlast,
                                            int *qcur, realtype *hinused, realtype *hlast,
                                            realtype *hcur, realtype *tcur)

    int CVodeGetNumNonlinSolvIters(void *cvode_mem, long int *nniters)
    int CVodeGetNumNonlinSolvConvFails(void *cvode_mem, long int *nncfails)
    int CVodeGetNonlinSolvStats(void *cvode_mem, long int *nniters,
                                            long int *nncfails)
    char *CVodeGetReturnFlagName(long int flag)
    void CVodeFree(void **cvode_mem)

    # Exported Functions -- Quadrature

    int CVodeQuadInit(void *cvode_mem, CVQuadRhsFn fQ, N_Vector yQ0)
    int CVodeQuadReInit(void *cvode_mem, N_Vector yQ0)
    int CVodeQuadSStolerances(void *cvode_mem, realtype reltolQ,
                              realtype abstolQ)
    int CVodeQuadSVtolerances(void *cvode_mem, realtype reltolQ,
                              N_Vector abstolQ)

    int CVodeSetQuadErrCon(void *cvode_mem, booleantype errconQ)

    int CVodeGetQuad(void *cvode_mem, realtype *tret, N_Vector yQout)
    int CVodeGetQuadDky(void *cvode_mem, realtype t, int k, N_Vector dky)

    int CVodeGetQuadNumRhsEvals(void *cvode_mem, long int *nfQevals)
    int CVodeGetQuadNumErrTestFails(void *cvode_mem, long int *nQetfails)
    int CVodeGetQuadErrWeights(void *cvode_mem, N_Vector eQweight)
    int CVodeGetQuadStats(void *cvode_mem, long int *nfQevals, 
                          long int *nQetfails)

    void CVodeQuadFree(void *cvode_mem)

    # Exported Functions -- Sensitivities
    int CVodeSensInit(void *cvode_mem, int Ns, int ism,
                      CVSensRhsFn fS, N_Vector *yS0)
    int CVodeSensInit1(void *cvode_mem, int Ns, int ism,
                       CVSensRhs1Fn fS1, N_Vector *yS0)
    int CVodeSensReInit(void *cvode_mem, int ism, N_Vector *yS0)

    int CVodeSensSStolerances(void *cvode_mem, realtype reltolS,
                              realtype *abstolS)
    int CVodeSensSVtolerances(void *cvode_mem, realtype reltolS,
                              N_Vector *abstolS)
    int CVodeSensEEtolerances(void *cvode_mem)

    int CVodeSetSensDQMethod(void *cvode_mem, int DQtype, realtype DQrhomax)
    int CVodeSetSensErrCon(void *cvode_mem, booleantype errconS)
    int CVodeSetSensMaxNonlinIters(void *cvode_mem, int maxcorS)
    int CVodeSetSensParams(void *cvode_mem, realtype *p,
                           realtype *pbar, int *plist)

    int CVodeSetNonlinearSolverSensSim(void *cvode_mem,
                                       SUNNonlinearSolver NLS)
    int CVodeSetNonlinearSolverSensStg(void *cvode_mem,
                                       SUNNonlinearSolver NLS)
    int CVodeSetNonlinearSolverSensStg1(void *cvode_mem,
                                        SUNNonlinearSolver NLS)

    int CVodeSensToggleOff(void *cvode_mem)

    int CVodeGetSens(void *cvode_mem, realtype *tret, N_Vector *ySout)
    int CVodeGetSens1(void *cvode_mem, realtype *tret, int iss,
                      N_Vector ySout) # rename is to iss, is protected in python
    int CVodeGetSensDky(void *cvode_mem, realtype t, int k,
                        N_Vector *dkyA)
    int CVodeGetSensDky1(void *cvode_mem, realtype t, int k, int iss,
                         N_Vector dky) # rename is to iss, is protected in python

    int CVodeGetSensNumRhsEvals(void *cvode_mem, long int *nfSevals)
    int CVodeGetNumRhsEvalsSens(void *cvode_mem, long int *nfevalsS)
    int CVodeGetSensNumErrTestFails(void *cvode_mem, long int *nSetfails)
    int CVodeGetSensNumLinSolvSetups(void *cvode_mem, long int *nlinsetupsS)
    int CVodeGetSensErrWeights(void *cvode_mem, N_Vector *eSweight)
    int CVodeGetSensStats(void *cvode_mem, long int *nfSevals,
                          long int *nfevalsS, long int *nSetfails,
                          long int *nlinsetupsS)
    int CVodeGetSensNumNonlinSolvIters(void *cvode_mem, long int *nSniters)
    int CVodeGetSensNumNonlinSolvConvFails(void *cvode_mem, long int *nSncfails)
    int CVodeGetStgrSensNumNonlinSolvIters(void *cvode_mem,
                                           long int *nSTGR1niters)
    int CVodeGetStgrSensNumNonlinSolvConvFails(void *cvode_mem,
                                               long int *nSTGR1ncfails)
    int CVodeGetSensNonlinSolvStats(void *cvode_mem, long int *nSniters,
                                    long int *nSncfails)

    void CVodeSensFree(void *cvode_mem)

    #Exported Functions -- Sensitivity dependent quadrature
    int CVodeQuadSensInit(void *cvode_mem, CVQuadSensRhsFn fQS,
                          N_Vector *yQS0)
    int CVodeQuadSensReInit(void *cvode_mem, N_Vector *yQS0)

    int CVodeQuadSensSStolerances(void *cvode_mem, realtype reltolQS,
                                  realtype *abstolQS)
    int CVodeQuadSensSVtolerances(void *cvode_mem, realtype reltolQS,
                                  N_Vector *abstolQS)
    int CVodeQuadSensEEtolerances(void *cvode_mem)

    int CVodeSetQuadSensErrCon(void *cvode_mem, booleantype errconQS)

    int CVodeGetQuadSens(void *cvode_mem, realtype *tret, N_Vector *yQSout)
    int CVodeGetQuadSens1(void *cvode_mem, realtype *tret, int iss,
                          N_Vector yQSout) # rename is to iss, is protected in python

    int CVodeGetQuadSensDky(void *cvode_mem, realtype t, int k,
                            N_Vector *dkyQS_all)
    int CVodeGetQuadSensDky1(void *cvode_mem, realtype t, int k,
                             int iss, N_Vector dkyQS) # rename is to iss, is protected in python

    int CVodeGetQuadSensNumRhsEvals(void *cvode_mem, long int *nfQSevals)
    int CVodeGetQuadSensNumErrTestFails(void *cvode_mem,
                                        long int *nQSetfails)
    int CVodeGetQuadSensErrWeights(void *cvode_mem, N_Vector *eQSweight)
    int CVodeGetQuadSensStats(void *cvode_mem, long int *nfQSevals,
                              long int *nQSetfails)

    void CVodeQuadSensFree(void *cvode_mem)

    # Exported Functions -- Backward Problems
    int CVodeAdjInit(void *cvode_mem, long int steps, int interp)
    int CVodeAdjReInit(void *cvode_mem)
    void CVodeAdjFree(void *cvode_mem)

    int CVodeCreateB(void *cvode_mem, int lmmB, int *which)
    int CVodeInitB(void *cvode_mem, int which, CVRhsFnB fB, realtype tB0, 
                   N_Vector yB0)
    int CVodeInitBS(void *cvode_mem, int which, CVRhsFnBS fBs,
                    realtype tB0, N_Vector yB0)
    int CVodeReInitB(void *cvode_mem, int which, realtype tB0, N_Vector yB0)
    int CVodeSStolerancesB(void *cvode_mem, int which, realtype reltolB,
                           realtype abstolB)
    int CVodeSVtolerancesB(void *cvode_mem, int which,
                           realtype reltolB, N_Vector abstolB)

    int CVodeQuadInitB(void *cvode_mem, int which,
                       CVQuadRhsFnB fQB, N_Vector yQB0)
    int CVodeQuadInitBS(void *cvode_mem, int which,
                        CVQuadRhsFnBS fQBs, N_Vector yQB0)
    int CVodeQuadReInitB(void *cvode_mem, int which, N_Vector yQB0)

    int CVodeQuadSStolerancesB(void *cvode_mem, int which,
                               realtype reltolQB, realtype abstolQB)
    int CVodeQuadSVtolerancesB(void *cvode_mem, int which,
                               realtype reltolQB, N_Vector abstolQB)

    # Solver Function For Forward Problems 
    int CVodeF(void *cvode_mem, realtype tout, N_Vector yout,
               realtype *tret, int itask, int *ncheckPtr)

    # Solver Function For Backward Problems
    int CVodeB(void *cvode_mem, realtype tBout, int itaskB)

    # Optional Input Functions For Adjoint Problems
    int CVodeSetAdjNoSensi(void *cvode_mem)
    
    int CVodeSetUserDataB(void *cvode_mem, int which, void *user_dataB)
    int CVodeSetMaxOrdB(void *cvode_mem, int which, int maxordB)
    int CVodeSetMaxNumStepsB(void *cvode_mem, int which, long int mxstepsB)
    int CVodeSetStabLimDetB(void *cvode_mem, int which, booleantype stldetB)
    int CVodeSetInitStepB(void *cvode_mem, int which, realtype hinB)
    int CVodeSetMinStepB(void *cvode_mem, int which, realtype hminB)
    int CVodeSetMaxStepB(void *cvode_mem, int which, realtype hmaxB)
    int CVodeSetConstraintsB(void *cvode_mem, int which, N_Vector constraintsB)
    int CVodeSetQuadErrConB(void *cvode_mem, int which, booleantype errconQB)

    int CVodeSetNonlinearSolverB(void *cvode_mem, int which,
                                 SUNNonlinearSolver NLS)

    # Extraction And Dense Output Functions For Backward Problems
    int CVodeGetB(void *cvode_mem, int which, realtype *tBret, N_Vector yB)
    int CVodeGetQuadB(void *cvode_mem, int which, realtype *tBret, N_Vector qB)

    # Optional Output Functions For Backward Problems 
    void *CVodeGetAdjCVodeBmem(void *cvode_mem, int which)
    int CVodeGetAdjY(void *cvode_mem, realtype t, N_Vector y)

    struct _CVadjCheckPointRec:
        void *my_addr
        void *next_addr
        realtype t0
        realtype t1
        long int nstep
        int order
        realtype step

    ctypedef _CVadjCheckPointRec CVadjCheckPointRec

    int CVodeGetAdjCheckPointsInfo(void *cvode_mem, CVadjCheckPointRec *ckpnt)

    int CVodeGetAdjDataPointHermite(void *cvode_mem, int which,
                                    realtype *t, N_Vector y, N_Vector yd)

    int CVodeGetAdjDataPointPolynomial(void *cvode_mem, int which,
                                       realtype *t, int *order, N_Vector y)


    int CVodeGetAdjCurrentCheckPoint(void *cvode_mem, void **addr)

    
cdef extern from "cvodes/cvodes_ls.h":
    #CVDLS return values
    enum: CVLS_SUCCESS         #  0
    enum: CVLS_MEM_NULL        # -1
    enum: CVLS_LMEM_NULL       # -2
    enum: CVLS_ILL_INPUT       # -3
    enum: CVLS_MEM_FAIL        # -4
    enum: CVLS_PMEM_NULL        # -5
    enum: CVLS_JACFUNC_UNRECVR  # -6
    enum: CVLS_JACFUNC_RECVR    # -7
    enum: CVLS_SUNMAT_FAIL      # -8
    enum: CVLS_SUNLS_FAIL       # -9

    enum: CVLS_NO_ADJ           # -101
    enum: CVLS_LMEMB_NULL       # -102
    
    ctypedef int (*CVLsJacFn)(realtype t, N_Vector y, N_Vector fy,
                              SUNMatrix Jac, void *user_data,
                              N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) except? -1

    ctypedef int (*CVLsPrecSetupFn)(realtype t, N_Vector y, N_Vector fy,
                                    booleantype jok, booleantype *jcurPtr,
                                    realtype gamma, void *user_data) except? -1

    ctypedef int (*CVLsPrecSolveFn)(realtype t, N_Vector y, N_Vector fy,
                                    N_Vector r, N_Vector z, realtype gamma,
                                    realtype delta, int lr, void *user_data) except? -1

    ctypedef int (*CVLsJacTimesSetupFn)(realtype t, N_Vector y,
                                       N_Vector fy, void *user_data) except? -1

    ctypedef int (*CVLsJacTimesVecFn)(N_Vector v, N_Vector Jv, realtype t,
                                      N_Vector y, N_Vector fy,
                                      void *user_data, N_Vector tmp) except? -1

    ctypedef int (*CVLsLinSysFn)(realtype t, N_Vector y, N_Vector fy, 
                                 SUNMatrix A, booleantype jok, 
                                 booleantype *jcur, realtype gamma,
                                 void *user_data, N_Vector tmp1, N_Vector tmp2,
                                 N_Vector tmp3)


    int CVodeSetLinearSolver(void *cvode_mem, SUNLinearSolver LS, SUNMatrix A)

    int CVodeSetJacFn(void *cvode_mem, CVLsJacFn jac)
    int CVodeSetMaxStepsBetweenJac(void *cvode_mem, long int msbj)
    int CVodeSetEpsLin(void *cvode_mem, realtype eplifac)
    int CVodeSetPreconditioner(void *cvode_mem, CVLsPrecSetupFn pset,
                               CVLsPrecSolveFn psolve)
    int CVodeSetJacTimes(void *cvode_mem, CVLsJacTimesSetupFn jtsetup,
                         CVLsJacTimesVecFn jtimes)
    int CVodeSetLinSysFn(void *cvode_mem, CVLsLinSysFn linsys)

    int CVodeGetLinWorkSpace(void *cvode_mem, long int *lenrwLS,
                             long int *leniwLS)
    int CVodeGetNumJacEvals(void *cvode_mem, long int *njevals)
    int CVodeGetNumPrecEvals(void *cvode_mem, long int *npevals)
    int CVodeGetNumPrecSolves(void *cvode_mem, long int *npsolves)
    int CVodeGetNumLinIters(void *cvode_mem, long int *nliters)
    int CVodeGetNumLinConvFails(void *cvode_mem, long int *nlcfails)
    int CVodeGetNumJTSetupEvals(void *cvode_mem, long int *njtsetups)
    int CVodeGetNumJtimesEvals(void *cvode_mem, long int *njvevals)
    int CVodeGetNumLinRhsEvals(void *cvode_mem, long int *nfevalsLS)
    int CVodeGetLastLinFlag(void *cvode_mem, long int *flag)
    char *CVodeGetLinReturnFlagName(long int flag)

    #  Backward problems
    ctypedef int (*CVLsJacFnB)(realtype t, N_Vector y, N_Vector yB,
                               N_Vector fyB, SUNMatrix JB,
                               void *user_dataB, N_Vector tmp1B,
                               N_Vector tmp2B, N_Vector tmp3B)

    ctypedef int (*CVLsJacFnBS)(realtype t, N_Vector y, N_Vector *yS,
                                N_Vector yB, N_Vector fyB, SUNMatrix JB,
                                void *user_dataB, N_Vector tmp1B,
                                N_Vector tmp2B, N_Vector tmp3B)

    ctypedef int (*CVLsPrecSetupFnB)(realtype t, N_Vector y, N_Vector yB,
                                     N_Vector fyB, booleantype jokB,
                                     booleantype *jcurPtrB,
                                     realtype gammaB, void *user_dataB)

    ctypedef int (*CVLsPrecSetupFnBS)(realtype t, N_Vector y,
                                      N_Vector *yS, N_Vector yB,
                                      N_Vector fyB, booleantype jokB,
                                      booleantype *jcurPtrB,
                                      realtype gammaB, void *user_dataB)

    ctypedef int (*CVLsPrecSolveFnB)(realtype t, N_Vector y, N_Vector yB,
                                     N_Vector fyB, N_Vector rB,
                                     N_Vector zB, realtype gammaB,
                                     realtype deltaB, int lrB,
                                     void *user_dataB)

    ctypedef int (*CVLsPrecSolveFnBS)(realtype t, N_Vector y, N_Vector *yS,
                                      N_Vector yB, N_Vector fyB,
                                      N_Vector rB, N_Vector zB,
                                      realtype gammaB, realtype deltaB,
                                      int lrB, void *user_dataB)

    ctypedef int (*CVLsJacTimesSetupFnB)(realtype t, N_Vector y, N_Vector yB,
                                         N_Vector fyB, void *jac_dataB)

    ctypedef int (*CVLsJacTimesSetupFnBS)(realtype t, N_Vector y,
                                          N_Vector *yS, N_Vector yB,
                                          N_Vector fyB, void *jac_dataB)

    ctypedef int (*CVLsJacTimesVecFnB)(N_Vector vB, N_Vector JvB, realtype t,
                                       N_Vector y, N_Vector yB, N_Vector fyB,
                                       void *jac_dataB, N_Vector tmpB)

    ctypedef int (*CVLsJacTimesVecFnBS)(N_Vector vB, N_Vector JvB,
                                        realtype t, N_Vector y, N_Vector *yS,
                                        N_Vector yB, N_Vector fyB,
                                        void *jac_dataB, N_Vector tmpB)

    ctypedef int (*CVLsLinSysFnB)(realtype t, N_Vector y, N_Vector yB, 
                                  N_Vector fyB, SUNMatrix AB, booleantype jokB,
                                  booleantype *jcurB, realtype gammaB, 
                                  void *user_dataB, N_Vector tmp1B,
                                  N_Vector tmp2B, N_Vector tmp3B)

    ctypedef int (*CVLsLinSysFnBS)(realtype t, N_Vector y, N_Vector* yS,
                                   N_Vector yB, N_Vector fyB, SUNMatrix AB,
                                   booleantype jokB, booleantype *jcurB,
                                   realtype gammaB, void *user_dataB, 
                                   N_Vector tmp1B,
                                   N_Vector tmp2B, N_Vector tmp3B)

    int CVodeSetLinearSolverB(void *cvode_mem, int which, SUNLinearSolver LS,
                              SUNMatrix A)

    int CVodeSetJacFnB(void *cvode_mem, int which, CVLsJacFnB jacB)
    int CVodeSetJacFnBS(void *cvode_mem, int which, CVLsJacFnBS jacBS)

    int CVodeSetEpsLinB(void *cvode_mem, int which, realtype eplifacB)

    int CVodeSetPreconditionerB(void *cvode_mem, int which, 
                                CVLsPrecSetupFnB psetB,
                                CVLsPrecSolveFnB psolveB)
    int CVodeSetPreconditionerBS(void *cvode_mem, int which,
                                 CVLsPrecSetupFnBS psetBS,
                                 CVLsPrecSolveFnBS psolveBS)

    int CVodeSetJacTimesB(void *cvode_mem, int which,
                          CVLsJacTimesSetupFnB jtsetupB,
                          CVLsJacTimesVecFnB jtimesB)
    int CVodeSetJacTimesBS(void *cvode_mem, int which,
                           CVLsJacTimesSetupFnBS jtsetupBS,
                           CVLsJacTimesVecFnBS jtimesBS)

    int CVodeSetLinSysFnB(void *cvode_mem, int which, CVLsLinSysFnB linsys)
    int CVodeSetLinSysFnBS(void *cvode_mem, int which, CVLsLinSysFnBS linsys)

cdef extern from "cvodes/cvodes_direct.h":
    ctypedef CVLsJacFn CVDlsJacFn
    ctypedef CVLsJacFnB CVDlsJacFnB
    ctypedef CVLsJacFnBS CVDlsJacFnBS

    int CVDlsSetLinearSolver(void *cvode_mem, SUNLinearSolver LS,
                             SUNMatrix A)
    int CVDlsSetJacFn(void *cvode_mem, CVDlsJacFn jac)
    int CVDlsGetWorkSpace(void *cvode_mem, long int *lenrwLS, long int *leniwLS)
    int CVDlsGetNumJacEvals(void *cvode_mem, long int *njevals)
    int CVDlsGetNumRhsEvals(void *cvode_mem, long int *nfevalsLS)
    int CVDlsGetLastFlag(void *cvode_mem, long int *flag)
    char *CVDlsGetReturnFlagName(long int flag)

    int CVDlsSetLinearSolverB(void *cvode_mem, int which,
                              SUNLinearSolver LS, SUNMatrix A)
    int CVDlsSetJacFnB(void *cvode_mem, int which, CVDlsJacFnB jacB)
    int CVDlsSetJacFnBS(void *cvode_mem, int which, CVDlsJacFnBS jacBS)

cdef extern from "cvodes/cvodes_bandpre.h":
    int CVBandPrecInit(void *cvode_mem, sunindextype N, sunindextype mu,
                       sunindextype ml);
    int CVBandPrecGetWorkSpace(void *cvode_mem, long int *lenrwLS, 
                               long int *leniwLS)
    int CVBandPrecGetNumRhsEvals(void *cvode_mem, long int *nfevalsBP)

    int CVBandPrecInitB(void *cvode_mem, int which,
                        sunindextype nB, sunindextype muB, sunindextype mlB)

cdef extern from "cvodes/cvodes_diag.h":
    # CVDIAG return values
    enum: CVDIAG_SUCCESS         #  0
    enum: CVDIAG_MEM_NULL        # -1
    enum: CVDIAG_LMEM_NULL       # -2
    enum: CVDIAG_ILL_INPUT       # -3
    enum: CVDIAG_MEM_FAIL        # -4
    # Additional last_flag values 
    enum: CVDIAG_INV_FAIL        # -5
    enum: CVDIAG_RHSFUNC_UNRECVR # -6
    enum: CVDIAG_RHSFUNC_RECVR   # -7

    enum: CVDIAG_NO_ADJ          # -101
    
    int CVDiag(void *cvode_mem)
    int CVDiagGetWorkSpace(void *cvode_mem, long int *lenrwLS, long int *leniwLS)
    int CVDiagGetNumRhsEvals(void *cvode_mem, long int *nfevalsLS)
    int CVDiagGetLastFlag(void *cvode_mem, long int *flag)
    char *CVDiagGetReturnFlagName(long int flag)

    int CVDiagB(void *cvode_mem, int which)

cdef extern from "cvodes/cvodes_bbdpre.h":
    ctypedef int (*CVLocalFn)(sunindextype Nlocal, realtype t, N_Vector y,
                              N_Vector g, void *user_data)
    ctypedef int (*CVCommFn)(sunindextype Nlocal, realtype t, N_Vector y,
                             void *user_data)

    int CVBBDPrecInit(void *cvode_mem, sunindextype Nlocal,
                      sunindextype mudq, sunindextype mldq,
                      sunindextype mukeep, sunindextype mlkeep,
                      realtype dqrely, CVLocalFn gloc, CVCommFn cfn)
    int CVBBDPrecReInit(void *cvode_mem, sunindextype mudq, sunindextype mldq,
                        realtype dqrely)
    int CVBBDPrecGetWorkSpace(void *cvode_mem, long int *lenrwBBDP, 
                              long int *leniwBBDP)
    int CVBBDPrecGetNumGfnEvals(void *cvode_mem, long int *ngevalsBBDP)

    ctypedef int (*CVLocalFnB)(sunindextype NlocalB, realtype t,
                               N_Vector y, N_Vector yB, N_Vector gB, 
                               void *user_dataB)

    ctypedef int (*CVCommFnB)(sunindextype NlocalB, realtype t,
                              N_Vector y, N_Vector yB, void *user_dataB)

    int CVBBDPrecInitB(void *cvode_mem, int which, sunindextype NlocalB,
                       sunindextype mudqB, sunindextype mldqB,
                       sunindextype mukeepB, sunindextype mlkeepB,
                       realtype dqrelyB, CVLocalFnB glocB, CVCommFnB cfnB)

    int CVBBDPrecReInitB(void *cvode_mem, int which,
                         sunindextype mudqB, sunindextype mldqB,
                         realtype dqrelyB)

cdef extern from "cvodes/cvodes_spils.h":

    ctypedef CVLsPrecSetupFn CVSpilsPrecSetupFn
    ctypedef CVLsPrecSolveFn CVSpilsPrecSolveFn
    ctypedef CVLsJacTimesSetupFn CVSpilsJacTimesSetupFn
    ctypedef CVLsJacTimesVecFn CVSpilsJacTimesVecFn
    
    ctypedef CVLsPrecSetupFnB CVSpilsPrecSetupFnB;
    ctypedef CVLsPrecSetupFnBS CVSpilsPrecSetupFnBS;
    ctypedef CVLsPrecSolveFnB CVSpilsPrecSolveFnB;
    ctypedef CVLsPrecSolveFnBS CVSpilsPrecSolveFnBS;
    ctypedef CVLsJacTimesSetupFnB CVSpilsJacTimesSetupFnB;
    ctypedef CVLsJacTimesSetupFnBS CVSpilsJacTimesSetupFnBS;
    ctypedef CVLsJacTimesVecFnB CVSpilsJacTimesVecFnB;
    ctypedef CVLsJacTimesVecFnBS CVSpilsJacTimesVecFnBS;

    int CVSpilsSetLinearSolver(void *cvode_mem, SUNLinearSolver LS)
    int CVSpilsSetEpsLin(void *cvode_mem, realtype eplifac)
    int CVSpilsSetPreconditioner(void *cvode_mem, CVSpilsPrecSetupFn pset,
                                 CVSpilsPrecSolveFn psolve)
    int CVSpilsSetJacTimes(void *cvode_mem, CVSpilsJacTimesSetupFn jtsetup,
                           CVSpilsJacTimesVecFn jtimes)

    int CVSpilsGetWorkSpace(void *cvode_mem, long int *lenrwLS, long int *leniwLS)
    int CVSpilsGetNumPrecEvals(void *cvode_mem, long int *npevals)
    int CVSpilsGetNumPrecSolves(void *cvode_mem, long int *npsolves)
    int CVSpilsGetNumLinIters(void *cvode_mem, long int *nliters)
    int CVSpilsGetNumConvFails(void *cvode_mem, long int *nlcfails)
    int CVSpilsGetNumJTSetupEvals(void *cvode_mem, long int *njtsetups)
    int CVSpilsGetNumJtimesEvals(void *cvode_mem, long int *njvevals)
    int CVSpilsGetNumRhsEvals(void *cvode_mem, long int *nfevalsLS)
    int CVSpilsGetLastFlag(void *cvode_mem, long int *flag)
    char *CVSpilsGetReturnFlagName(long int flag)

    int CVSpilsSetLinearSolverB(void *cvode_mem, int which, SUNLinearSolver LS)
    int CVSpilsSetEpsLinB(void *cvode_mem, int which, realtype eplifacB)
    int CVSpilsSetPreconditionerB(void *cvode_mem, int which,
                                  CVSpilsPrecSetupFnB psetB,
                                  CVSpilsPrecSolveFnB psolveB)
    int CVSpilsSetPreconditionerBS(void *cvode_mem, int which,
                                   CVSpilsPrecSetupFnBS psetBS,
                                   CVSpilsPrecSolveFnBS psolveBS)
    int CVSpilsSetJacTimesB(void *cvode_mem, int which,
                            CVSpilsJacTimesSetupFnB jtsetupB,
                            CVSpilsJacTimesVecFnB jtimesB)
    int CVSpilsSetJacTimesBS(void *cvode_mem, int which,
                             CVSpilsJacTimesSetupFnBS jtsetupBS,
                             CVSpilsJacTimesVecFnBS jtimesBS)
