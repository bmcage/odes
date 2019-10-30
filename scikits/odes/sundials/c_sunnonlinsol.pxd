from .c_sundials cimport *
from .c_sunmatrix cimport *

include "sundials_config.pxi"


cdef extern from "sunnonlinsol/sunnonlinsol_newton.h":

    struct _SUNNonlinearSolverContent_Newton:
    
      SUNNonlinSolSysFn      Sys
      SUNNonlinSolLSetupFn   LSetup
      SUNNonlinSolLSolveFn   LSolve
      SUNNonlinSolConvTestFn CTest
    
      N_Vector    delta
      booleantype jcur
      int         curiter
      int         maxiters
      long int    niters
      long int    nconvfails
    
    ctypedef _SUNNonlinearSolverContent_Newton *SUNNonlinearSolverContent_Newton;

    SUNNonlinearSolver SUNNonlinSol_Newton(N_Vector y)
    SUNNonlinearSolver SUNNonlinSol_NewtonSens(int count, N_Vector y)

    SUNNonlinearSolver_Type SUNNonlinSolGetType_Newton(SUNNonlinearSolver NLS)

    int SUNNonlinSolInitialize_Newton(SUNNonlinearSolver NLS)
    int SUNNonlinSolSolve_Newton(SUNNonlinearSolver NLS,  N_Vector y0, 
                                 N_Vector y, N_Vector w, realtype tol,
                                 booleantype callLSetup, void *mem)
    int SUNNonlinSolFree_Newton(SUNNonlinearSolver NLS)

    int SUNNonlinSolSetSysFn_Newton(SUNNonlinearSolver NLS,
                                    SUNNonlinSolSysFn SysFn);
    int SUNNonlinSolSetLSetupFn_Newton(SUNNonlinearSolver NLS,
                                       SUNNonlinSolLSetupFn LSetupFn);
    int SUNNonlinSolSetLSolveFn_Newton(SUNNonlinearSolver NLS,
                                       SUNNonlinSolLSolveFn LSolveFn);
    int SUNNonlinSolSetConvTestFn_Newton(SUNNonlinearSolver NLS,
                                         SUNNonlinSolConvTestFn CTestFn);
    int SUNNonlinSolSetMaxIters_Newton(SUNNonlinearSolver NLS,
                                       int maxiters);

    int SUNNonlinSolGetNumIters_Newton(SUNNonlinearSolver NLS,
                                       long int *niters);
    int SUNNonlinSolGetCurIter_Newton(SUNNonlinearSolver NLS, int *iter);
    int SUNNonlinSolGetNumConvFails_Newton(SUNNonlinearSolver NLS,
                                           long int *nconvfails);
    int SUNNonlinSolGetSysFn_Newton(SUNNonlinearSolver NLS,
                                    SUNNonlinSolSysFn *SysFn);


cdef extern from "sunnonlinsol/sunnonlinsol_fixedpoint.h":
    
    struct _SUNNonlinearSolverContent_FixedPoint:

        SUNNonlinSolSysFn      Sys
        SUNNonlinSolConvTestFn CTest

        int       m
        int      *imap
        realtype *R
        realtype *gamma
        realtype *cvals
        N_Vector *df
        N_Vector *dg
        N_Vector *q
        N_Vector *Xvecs
        N_Vector  yprev
        N_Vector  gy
        N_Vector  fold
        N_Vector  gold
        N_Vector  delta
        int       curiter
        int       maxiters
        long int  niters
        long int  nconvfails

    ctypedef _SUNNonlinearSolverContent_FixedPoint *SUNNonlinearSolverContent_FixedPoint

    SUNNonlinearSolver SUNNonlinSol_FixedPoint(N_Vector y, int m)
    SUNNonlinearSolver SUNNonlinSol_FixedPointSens(int count, N_Vector y, int m)

    SUNNonlinearSolver_Type SUNNonlinSolGetType_FixedPoint(SUNNonlinearSolver NLS)

    int SUNNonlinSolInitialize_FixedPoint(SUNNonlinearSolver NLS)
    int SUNNonlinSolSolve_FixedPoint(SUNNonlinearSolver NLS,
                                     N_Vector y0, N_Vector y,
                                     N_Vector w, realtype tol,
                                     booleantype callSetup, void *mem)
    int SUNNonlinSolFree_FixedPoint(SUNNonlinearSolver NLS)

    int SUNNonlinSolSetSysFn_FixedPoint(SUNNonlinearSolver NLS,
                                        SUNNonlinSolSysFn SysFn);
    int SUNNonlinSolSetConvTestFn_FixedPoint(SUNNonlinearSolver NLS,
                                             SUNNonlinSolConvTestFn CTestFn);
    int SUNNonlinSolSetMaxIters_FixedPoint(SUNNonlinearSolver NLS,
                                           int maxiters);

    int SUNNonlinSolGetNumIters_FixedPoint(SUNNonlinearSolver NLS,
                                           long int *niters)
    int SUNNonlinSolGetCurIter_FixedPoint(SUNNonlinearSolver NLS, int *iter)
    int SUNNonlinSolGetNumConvFails_FixedPoint(SUNNonlinearSolver NLS,
                                               long int *nconvfails)
    int SUNNonlinSolGetSysFn_FixedPoint(SUNNonlinearSolver NLS,
                                        SUNNonlinSolSysFn *SysFn)

