from .c_sundials cimport *
from libc.stdio cimport FILE

cdef extern from "cvode/cvode.h":
    # lmm
    enum: CV_ADAMS # 1
    enum: CV_BDF   # 2

    # itask
    enum: CV_NORMAL     # 1
    enum: CV_ONE_STEP   # 2

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
    
    enum: CV_UNRECOGNIZED_ERR  #   -99

    ctypedef int (*CVRhsFn)(realtype t, N_Vector y, N_Vector ydot, void *user_data) except? -1
    ctypedef int (*CVRootFn)(realtype t, N_Vector y, realtype *gout, void *user_data) except? -1
    ctypedef int (*CVEwtFn)(N_Vector y, N_Vector ewt, void *user_data)
    ctypedef void (*CVErrHandlerFn)(int error_code,
                               char *module, char *function,
                               char *msg, void *user_data)

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

cdef extern from "cvode/cvode_ls.h":
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

cdef extern from "cvode/cvode_direct.h":
    ctypedef CVLsJacFn CVDlsJacFn

    int CVDlsSetLinearSolver(void *cvode_mem, SUNLinearSolver LS,
                             SUNMatrix A)
    int CVDlsSetJacFn(void *cvode_mem, CVDlsJacFn jac)
    int CVDlsGetWorkSpace(void *cvode_mem, long int *lenrwLS, long int *leniwLS)
    int CVDlsGetNumJacEvals(void *cvode_mem, long int *njevals)
    int CVDlsGetNumRhsEvals(void *cvode_mem, long int *nfevalsLS)
    int CVDlsGetLastFlag(void *cvode_mem, long int *flag)
    char *CVDlsGetReturnFlagName(long int flag)

cdef extern from "cvode/cvode_bandpre.h":
    int CVBandPrecInit(void *cvode_mem, sunindextype N, sunindextype mu,
                       sunindextype ml);
    int CVBandPrecGetWorkSpace(void *cvode_mem, long int *lenrwLS, 
                               long int *leniwLS)
    int CVBandPrecGetNumRhsEvals(void *cvode_mem, long int *nfevalsBP)

cdef extern from "cvode/cvode_diag.h":
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

    int CVDiag(void *cvode_mem)
    int CVDiagGetWorkSpace(void *cvode_mem, long int *lenrwLS, long int *leniwLS)
    int CVDiagGetNumRhsEvals(void *cvode_mem, long int *nfevalsLS)
    int CVDiagGetLastFlag(void *cvode_mem, long int *flag)
    char *CVDiagGetReturnFlagName(long int flag)

cdef extern from "cvode/cvode_bbdpre.h":
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

cdef extern from "cvode/cvode_spils.h":

    ctypedef CVLsPrecSetupFn CVSpilsPrecSetupFn
    ctypedef CVLsPrecSolveFn CVSpilsPrecSolveFn
    ctypedef CVLsJacTimesSetupFn CVSpilsJacTimesSetupFn
    ctypedef CVLsJacTimesVecFn CVSpilsJacTimesVecFn

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
