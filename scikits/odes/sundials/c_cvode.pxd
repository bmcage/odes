from .c_sundials cimport *
from libc.stdio cimport FILE

cdef extern from "cvode/cvode.h":
    # lmm
    enum: CV_ADAMS # 1
    enum: CV_BDF   # 2

    # iter
    enum: CV_FUNCTIONAL # 1
    enum: CV_NEWTON     # 2

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

    enum: CV_MEM_FAIL          #   -20
    enum: CV_MEM_NULL          #   -21
    enum: CV_ILL_INPUT         #   -22
    enum: CV_NO_MALLOC         #   -23
    enum: CV_BAD_K             #   -24
    enum: CV_BAD_T             #   -25
    enum: CV_BAD_DKY           #   -26
    enum: CV_TOO_CLOSE         #   -27

    ctypedef int (*CVRhsFn)(realtype t, N_Vector y, N_Vector ydot, void *user_data) except? -1
    ctypedef int (*CVRootFn)(realtype t, N_Vector y, realtype *gout, void *user_data) except? -1
    ctypedef int (*CVEwtFn)(N_Vector y, N_Vector ewt, void *user_data)
    ctypedef void (*CVErrHandlerFn)(int error_code,
                               char *module, char *function,
                               char *msg, void *user_data)

    void *CVodeCreate(int lmm, int iter)
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

    int CVodeSetIterType(void *cvode_mem, int iter)

    int CVodeSetRootDirection(void *cvode_mem, int *rootdir)
    int CVodeSetNoInactiveRootWarn(void *cvode_mem)

    int CVodeInit(void *cvode_mem, CVRhsFn f, realtype t0, N_Vector y0)
    int CVodeReInit(void *cvode_mem, realtype t0, N_Vector y0)

    int CVodeSStolerances(void *cvode_mem, realtype reltol, realtype abstol)
    int CVodeSVtolerances(void *cvode_mem, realtype reltol, N_Vector abstol)
    int CVodeWFtolerances(void *cvode_mem, CVEwtFn efun)
    int CVodeRootInit(void *cvode_mem, int nrtfn, CVRootFn g)
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
    int CVodeGetNumStabLimOrderReds(void *cvode_mem, long int *nslred)
    int CVodeGetActualInitStep(void *cvode_mem, realtype *hinused)
    int CVodeGetLastStep(void *cvode_mem, realtype *hlast)
    int CVodeGetCurrentStep(void *cvode_mem, realtype *hcur)
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

cdef extern from "cvode/cvode_direct.h":
    #CVDLS return values
    enum: CVDLS_SUCCESS         #  0
    enum: CVDLS_MEM_NULL        # -1
    enum: CVDLS_LMEM_NULL       # -2
    enum: CVDLS_ILL_INPUT       # -3
    enum: CVDLS_MEM_FAIL        # -4

    # Additional last_flag values

    enum: CVDLS_JACFUNC_UNRECVR # -5
    enum: CVDLS_JACFUNC_RECVR   # -6
    enum: CVDLS_SUNMAT_FAIL     # -7

    ctypedef int (*CVDlsJacFn)(realtype t, N_Vector y, N_Vector fy,
                          SUNMatrix Jac, void *user_data,
                          N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) except? -1

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
    int CVBandPrecGetWorkSpace(void *cvode_mem, long int *lenrwLS, long int *leniwLS)
    int CVBandPrecGetNumRhsEvals(void *cvode_mem, long int *nfevalsBP)

cdef extern from "cvode/cvode_diag.h":
    int CVDiag(void *cvode_mem)
    int CVDiagGetWorkSpace(void *cvode_mem, long int *lenrwLS, long int *leniwLS)
    int CVDiagGetNumRhsEvals(void *cvode_mem, long int *nfevalsLS)
    int CVDiagGetLastFlag(void *cvode_mem, long int *flag)
    char *CVDiagGetReturnFlagName(long int flag)

    # CVDIAG return values
    enum: CVDIAG_SUCCESS         #  0
    enum: CVDIAG_MEM_NULL        # -1
    enum: CVDIAG_LMEM_NULL       # -2
    enum: CVDIAG_ILL_INPUT       # -3
    enum: CVDIAG_MEM_FAIL        # -4
    # Additional last_flag values */
    enum: CVDIAG_INV_FAIL        # -5
    enum: CVDIAG_RHSFUNC_UNRECVR # -6
    enum: CVDIAG_RHSFUNC_RECVR   # -7

cdef extern from "cvode/cvode_bbdpre.h":
    ctypedef int (*CVLocalFn)(sunindextype Nlocal, realtype t, N_Vector y,
                         N_Vector g, void *user_data)
    ctypedef int (*CVCommFn)(sunindextype Nlocal, realtype t, N_Vector y,
                        void *user_data)

    int CVBBDPrecInit(void *cvode_mem, sunindextype Nlocal,
                                  sunindextype mudq, sunindextype mldq,
                                  sunindextype mukeep, sunindextype mlkeep,
                                  realtype dqrely,
                                  CVLocalFn gloc, CVCommFn cfn)
    int CVBBDPrecReInit(void *cvode_mem, sunindextype mudq, sunindextype mldq,
                                    realtype dqrely)
    int CVBBDPrecGetWorkSpace(void *cvode_mem, long int *lenrwLS, long int *leniwLS)
    int CVBBDPrecGetNumGfnEvals(void *cvode_mem, long int *ngevalsBBDP)

cdef extern from "cvode/cvode_spils.h":
    # CVSPILS return values
    enum: CVSPILS_SUCCESS        #  0
    enum: CVSPILS_MEM_NULL       # -1
    enum: CVSPILS_LMEM_NULL      # -2
    enum: CVSPILS_ILL_INPUT      # -3
    enum: CVSPILS_MEM_FAIL       # -4
    enum: CVSPILS_PMEM_NULL      # -5
    enum: CVSPILS_SUNLS_FAIL     # -6

    enum: CVSPILS_MSBPRE # 50
    enum: CVSPILS_DGMAX  # RCONST(0.2)
    enum: CVSPILS_EPLIN  # RCONST(0.05)

    ctypedef int (*CVSpilsPrecSetupFn)(realtype t, N_Vector y, N_Vector fy,
                                  booleantype jok, booleantype *jcurPtr,
                                  realtype gamma, void *user_data) except? -1
    ctypedef int (*CVSpilsPrecSolveFn)(realtype t, N_Vector y, N_Vector fy,
                                  N_Vector r, N_Vector z,
                                  realtype gamma, realtype delta,
                                  int lr, void *user_data) except? -1
    ctypedef int (*CVSpilsJacTimesSetupFn)(realtype t, N_Vector y,
                                      N_Vector fy, void *user_data) except? -1
    ctypedef int (*CVSpilsJacTimesVecFn)(N_Vector v, N_Vector Jv, realtype t,
                                    N_Vector y, N_Vector fy,
                                    void *user_data, N_Vector tmp) except? -1

    int CVSpilsSetLinearSolver(void *cvode_mem, SUNLinearSolver LS)
    int CVSpilsSetEpsLin(void *cvode_mem, realtype eplifac)
    int CVSpilsSetPreconditioner(void *cvode_mem,
                                             CVSpilsPrecSetupFn pset,
                                             CVSpilsPrecSolveFn psolve)
    int CVSpilsSetJacTimes(void *cvode_mem,
                                       CVSpilsJacTimesSetupFn jtsetup,
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
