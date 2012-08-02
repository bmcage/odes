import numpy as np
cimport numpy as np

from c_sundials cimport realtype, N_Vector
from c_cvode cimport *
from common_defs cimport (nv_s2ndarray, ndarray2nv_s,
                          ndarray2DlsMatd,
                          CV_RhsFunction, CV_WrapRhsFunction,
                          CV_RootFunction, CV_WrapRootFunction,
                          CV_JacRhsFunction, CV_WrapJacRhsFunction)

# TODO: parallel implementation: N_VectorParallel
# TODO: linsolvers: check the output value for errors
# TODO: unify using float/double/realtype variable
# TODO: optimize code for compiler

cdef int _rhsfn(realtype tt, N_Vector yy, N_Vector yp,
              void *auxiliary_data):
    """ function with the signature of CVRhsFn, that calls python Rhs """
    cdef np.ndarray[DTYPE_t, ndim=1] yy_tmp, yp_tmp

    aux_data = <CV_data> auxiliary_data
    cdef bint parallel_implementation = aux_data.parallel_implementation

    if parallel_implementation:
        raise NotImplemented
    else:
        yy_tmp = aux_data.yy_tmp
        yp_tmp = aux_data.yp_tmp

        nv_s2ndarray(yy, yy_tmp)
        #nv_s2ndarray(yp, yp_tmp)

    aux_data.rfn.evaluate(tt, yy_tmp, yp_tmp, aux_data.user_data)

    if parallel_implementation:
        raise NotImplemented
    else:
        ndarray2nv_s(yp, yp_tmp)

    return 0

cdef int _rootfn(realtype t, N_Vector y, realtype *gout, void *auxiliary_data):
    """ function with the signature of CVRootFn """

    aux_data = <CV_data> auxiliary_data
    cdef bint parallel_implementation = aux_data.parallel_implementation

    if parallel_implementation:
        raise NotImplemented
    else:
        yy_tmp = aux_data.yy_tmp
        g_tmp  = aux_data.g_tmp

        nv_s2ndarray(y, yy_tmp)

    aux_data.rootfn.evaluate(t, yy_tmp, g_tmp, aux_data.user_data)

    cdef int i
    if parallel_implementation:
        raise NotImplemented
    else:
        for i in np.arange(np.alen(g_tmp)):
            gout[i] = <realtype> g_tmp[i]

    return 0

cdef int _jacdense(int Neq, realtype tt,
            N_Vector yy, N_Vector ff, DlsMat Jac,
            void *auxiliary_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3):
    """function with the signature of CVDlsDenseJacFn that calls python Jac"""
    cdef np.ndarray[DTYPE_t, ndim=1] yy_tmp
    cdef np.ndarray jac_tmp

    aux_data = <CV_data> auxiliary_data
    cdef bint parallel_implementation = aux_data.parallel_implementation
    if parallel_implementation:
        raise NotImplemented
    else:
        yy_tmp = aux_data.yy_tmp
        if aux_data.jac_tmp == None:
            N = np.alen(yy_tmp)
            aux_data.jac_tmp = np.empty((N,N), float)
        jac_tmp = aux_data.jac_tmp

        nv_s2ndarray(yy, yy_tmp)
    aux_data.jac.evaluate(tt, yy_tmp, jac_tmp)

    if parallel_implementation:
        raise NotImplemented
    else:
        #we convert the python jac_tmp array to DslMat of sundials
        ndarray2DlsMatd(Jac, jac_tmp)

    return 0

cdef class CV_data:
    def __cinit__(self, N):
        self.parallel_implementation = False
        self.user_data = None

        self.yy_tmp = np.empty(N, float)
        self.yp_tmp = np.empty(N, float)
        self.jac_tmp = None
        self.g_tmp = None

cdef class CVODE:

    def __cinit__(self, Rfn, **options):
        """
        Initialize the CVODE Solver and it's default values

        Input:
            Rfn     - right-hand-side function
            options - additional options for initialization

        """

        default_values = {
            'implementation': 'serial',
            'lmm_type': 'BDF',
            'iter_type': 'NEWTON',
            'rtol': 1e-6, 'atol': 1e-12,
            'linsolver': 'dense',
            'lband': 0,'uband': 0,
            'maxl': 0,
            'precond_type': 'NONE',
            'tcrit': 0.,
            'order': 0,
            'max_step_size': 0.,
            'min_step_size': 0.,
            'first_step_size': 0.,
            'max_steps': 0,
            'bdf_stability_detection': False,
            'max_conv_fails': 0,
            'max_nonlin_iters': 0,
            'nonlin_conv_coef': 0.,
            'user_data': None,
            'rfn': None,
            'rootfn': None,
            'nr_rootfns': 0,
            'jacfn': None
            }

        self.options = default_values
        self.N       = -1
        self.set_options(rfn=Rfn, **options)
        self._cv_mem = NULL
        self.initialized = False

    def set_options(self, **options):
        """
        Reads the options list and assigns values for the solver.

        All options list:
            'implementation':
                Values: 'serial' (= default), 'parallel'
                Description:
                    Using serial or parallel implementation of the solver.
                    #TODO: curently only serial implementation is working
            'lmm_type'  - linear multistep method
                Values: 'ADAMS', 'BDF' (= default)
                Description:
                    Recommended combination:

                    Problem  | 'lmm_type' | 'iter_type'
                    ------------------------------------
                    nonstiff | 'ADAMS'    | 'FUNCTIONAL'
                    stiff    | 'BDF'      | 'NEWTON'

                    See also 'iter_type'.
            'iter_type' - nonlinear solver iteration
                Values: 'NEWTON' (= default), 'FUNCTIONAL'
                Description:
                    See 'lmm_type'.
            'rfn':
                Values: function of class CV_RhsFunction or a python function
                        with signature (t, y, yp) or (t, y, yp, userdata)
                Description:
                    Defines the right-hand-side function (which has to be
                    a subclass of CV_RhsFunction class, or a normal python
                    function with signature (t, y, yp) or (t, y, yp, userdata)).
                    This function takes as input arguments current time t,
                    current value of y, and yp must be used as output numpy
                    array of returned values of \dot{y} at t.
                    Optional userdata. Return value is 0 if successfull.
                    This option is mandatory.
            'rootfn':
                Values: function of class CV_RootFunction or a python function
                    with signature (t, y, g, user_data)
                Description:
                    Defines a function that fills a vector 'g' with values
                    (g is a vector of size k). Vector 'g represents conditions
                    which when satisfied, the solver stops. Namely, the solver
                    stops when condition g[i] is zero for some i.
                    For 'user_data' argument see 'user_data' option
                    above.
            'nr_rootfns':
                Value: integer
                Description:
                    The length of the array returned by 'rootfn' (see above),
                    i.e. number of conditions for which we search the value 0.
            'jacfn':
                Values: function of class CV_JacRhsFunction
                Description:
                    Defines the jacobian function and has to be a subclass
                    of CV_JacRhsFunction class or python function. This function
                    takes as input arguments current time t, current value of y,
                    a 2D numpy array of returned jacobian and optional userdata.
                    Return value is 0 if successfull.
                    Jacobian is only used for dense or lapackdense linear solver
            'rtol':
                Values: float,  1e-6 = default
                Description:
                    Relative tolerancy.
            'atol':
                Values: float or numpy array of floats,  1e-12 = default
                Description:
                    Absolute tolerancy
            'order':
                Values: 1, 2, 3, 4, 5 (= default)
                Description:
                    Specifies the maximum order of the linear multistep method.
            'max_steps':
                Values: positive integer, 0 = default (uses value of 500)
                Description:
                    Maximum number of (internally defined) steps allowed
                    during one call to the solver.
            'max_step_size':
                Values: non-negative float, 0.0 = default
                Description:
                    Restricts the maximal (absolute) internal step value taken
                    by the solver. The value of 0.0 uses
                    no restriction on the maximal step size.
            'min_step_size':
                Values: non-negative float, 0.0 = default
                Description:
                    Restricts the minimal (absolute) internal step value taken
                    by the solver. The value of 0.0 uses
                    no restriction.
            'first_step_size':
                Values: float , 0.0 = default
                Description:
                    Sets the first step size. DAE solver can suffer on the
                    first step, so set this to circumvent this. The value
                    of 0.0 uses the solver's internal default value.
            'linsolver':
                Values: 'dense' (= default), 'lapackdense', 'band',
                        'lapackband', 'spgmr', 'spbcg', 'sptfqmr'
                Description:
                    Specifies used linear solver.
                    Limitations: Linear solvers for dense and band matrices
                                 can be used only for serial implementation.
                                 For parallel implementation use_relaxation
                                 use lapackdense or lapackband respectively.
            'lband', 'uband':
                Values: non-negative integer, 0 = default
                Description:
                    Specifies the width of the lower band (below the main
                    diagonal) and/or upper diagonal (above the main diagonal).
                    So the total band width is lband + 1 + uband, where the 1
                    stands for the main diagonal
                    (specially, lband = uband = 0 [that is the default]
                    denotes the band width = 1, i.e. the main diagonal only).
                    Used only if 'linsolver' is band.
            'maxl':
                Values: 0 (= default), 1, 2, 3, 4, 5
                Description:
                    Dimension of the number of used Krylov subspaces
                    (used only by 'spgmr', 'spbcg', 'sptfqmr' linsolvers)
            'tcrit':
                Values: float, 0.0 = default
                Description:
                    Maximum time until which we perform the computations.
                    Unconstrained (i.e. infinite time) is value of 0.0.
            'user_data':
                Values: python object, None = default
                Description:
                    Additional data that are supplied to each call of the
                    residual function 'Rfn' (see below) and Jacobi function
                    'jacfn' (if specified one).
            'precond_type':
                default = None
            'bdf_stability_detection':
                default = False, only used if lmm_type == 'bdf
            'max_conv_fails':
                default = 0,
            'max_nonlin_iters':
                default = 0,
            'nonlin_conv_coef':
                default = 0.
        """

        for (key, value) in options.items():
            if key.lower() in self.options:
                self.options[key.lower()] = value
            else:
                raise ValueError("Option '%s' is not supported by solver" % key)

    cpdef _set_runtime_changeable_options(self, object options,
                                          bint supress_supported_check=False):
        """
          (Re)Set options that can change after CV_MEM object was allocated -
          mostly useful for values that need to be changed at run-time.
        """
        cdef int flag
        cdef void* cv_mem = self._cv_mem

        if cv_mem is NULL:
            return 0

        # Check whether options are all only supported - this check can
        # be supressed by 'supress_supported_check = True'
        if not supress_supported_check:
            for opt in options.keys():
                if not opt in ['atol', 'rtol', 'tcrit']:
                    raise ValueError("Option '%s' can''t be set runtime." % opt)

        # Set tolerances
        cdef N_Vector atol
        cdef np.ndarray[DTYPE_t, ndim=1] np_atol

        if 'atol' in options:
            opts_atol = options['atol']
        else:
            opts_atol = None

        if 'rtol' in options:
            opts_rtol = options['rtol']
        else:
            opts_rtol = None

        if not ((opts_rtol is None) and (opts_atol is None)):
            if opts_rtol is None:
                opts_rtol = self.options['rtol']
            else:
                if not np.isscalar(options['rtol']) :
                    raise ValueError("For IDA solver 'rtol' must be a scalar")

            if opts_atol is None:
                opts_atol = self.options['atol']
            else:
                if not (self.atol is NULL):
                    N_VDestroy(self.atol)
                    self.atol = NULL

            if np.isscalar(opts_atol):
                flag = CVodeSStolerances(cv_mem, <realtype> opts_rtol,
                                                 <realtype> opts_atol)
            else:
                np_atol = np.asarray(opts_atol)
                if np.alen(np_atol) != self.N:
                    raise ValueError("Array length inconsistency: 'atol' "
                                     "lenght (%i) differs from problem size "
                                     "(%i)." % (np.alen(np_atol), self.N))

                if self.parallel_implementation:
                    raise NotImplemented
                else:
                    atol = N_VMake_Serial(self.N, <realtype *> np_atol.data)
                    flag = CVodeSVtolerances(cv_mem, <realtype> opts_rtol, atol)

                    self.atol = atol

                if flag == CV_ILL_INPUT:
                    raise ValueError("CVodeStolerances: negative 'atol' or 'rtol' value.")
            #TODO: implement CVFWtolerances(cv_mem, efun)

        # Set tcrit
         if ('tcrit' in options):
            opts_tcrit = options['tcrit']
            if (not opts_tcrit is None) and (opts_tcrit >= 0.):
                flag = CVodeSetStopTime(cv_mem, <realtype> opts_tcrit)
                if flag == CV_ILL_INPUT:
                    raise ValueError('IDASetStopTime::Stop value is beyond '
                                     'current value.')

    def init_step(self, double t0, object y0):
        cdef np.ndarray[DTYPE_t, ndim=1] np_y0
        np_y0 = np.asarray(y0)

        return self._init_step(t0, np_y0)

    cpdef _init_step(self, DTYPE_t t0,
                     np.ndarray[DTYPE_t, ndim=1] y0):

        cdef dict opts = self.options

        lmm_type  = opts['lmm_type'].lower()
        iter_type = opts['iter_type'].lower()

        if lmm_type == 'bdf':
            lmm = CV_BDF
        elif lmm_type == 'adams':
            lmm = CV_ADAMS
        else:
            raise ValueError('CVODE:init: Unrecognized lmm_type: %s' % lmm_type)

        if iter_type == 'functional':
            itert = CV_FUNCTIONAL
        elif iter_type == 'newton':
            itert = CV_NEWTON
        else:
            raise ValueError('CVODE:init: Unrecognized iter_type: %s'
                             % iter_type)

        self.parallel_implementation = \
          (opts['implementation'].lower() == 'parallel')
        if self.parallel_implementation:
            raise ValueError('Error: Parallel implementation not implemented !')
        cdef long int N
        N = <long int> np.alen(y0)

        if opts['rfn'] == None:
            raise ValueError('The right-hand-side function rfn not assigned '
                              'during ''set_options'' call !')

        #TODO: when implementing parallel, does N_VDestroy be called separately
        #      for parallel version or it's a generic one?
        if not self.y0 is NULL:
            N_VDestroy(self.y0)
            N_VDestroy(self.y)
            N_VDestroy(self.yp)

        if self.parallel_implementation:
            raise NotImplemented
        else:
            self.y0  = N_VMake_Serial(N, <realtype *>y0.data)
            self.y   = N_VClone(self.y0)
            self.yp = N_VNew_Serial(N)

        cdef int flag
        cdef void* cv_mem = self._cv_mem

        if (cv_mem is NULL) or (self.N != N):
            if (not cv_mem is NULL):
                CVodeFree(&cv_mem)
            cv_mem = CVodeCreate(lmm, itert)
            if cv_mem is NULL:
                raise MemoryError('CVodeCreate:MemoryError: Could not create '
                                  'cv_mem object')

            self._cv_mem = cv_mem
            flag = CVodeInit(cv_mem, _rhsfn,  <realtype> t0, self.y0)
        elif self.N == N:
            flag = CVodeReInit(cv_mem, <realtype> t0, self.y0)
        else:
            raise ValueError('CVodeInit:Error: You should not be here...')
        if flag == CV_ILL_INPUT:
                raise ValueError('CVode[Re]Init: Ill input')
        elif flag == CV_MEM_FAIL:
            raise MemoryError('CV[Re]Init: Memory allocation error')
        elif flag == CV_MEM_NULL:
            raise MemoryError('CVodeCreate: Memory allocation error')
        elif flag == CV_NO_MALLOC:
            raise MemoryError('CVodeReInit: No memory allocated in CVInit.')

        self.N = N

        # auxiliary variables
        self.aux_data = CV_data(N)
        self.aux_data.parallel_implementation = self.parallel_implementation

        rfn = opts['rfn']
        if not isinstance(rfn , CV_RhsFunction):
            tmpfun = CV_WrapRhsFunction()
            tmpfun.set_rhsfn(rfn)
            rfn = tmpfun
            opts['rfn'] = tmpfun
        self.aux_data.rfn = rfn

        rootfn = opts['rootfn']
        if rootfn is not None:
            nr_rootfns = opts['nr_rootfns']
            if nr_rootfns is None:
                raise ValueError('Number of root-ing functions ''nr_rootfns'' '
                                 'must be specified.')
            if not isinstance(rootfn, CV_RootFunction):
                tmpfun = CV_WrapRootFunction()
                tmpfun.set_rootfn(rootfn)
                rootfn = tmpfun
                opts['rootfn'] = tmpfun

            self.aux_data.rootfn = rootfn
            self.aux_data.g_tmp  = np.empty([nr_rootfns,], float)

            flag = CVodeRootInit(cv_mem, nr_rootfns, _rootfn)

            if flag == CV_SUCCESS:
                pass
            if flag == CV_ILL_INPUT:
                raise ValueError('CVRootInit: Function ''rootfn'' is NULL '
                                 'but ''nr_rootfns'' > 0')
            elif flag == CV_MEM_FAIL:
                raise MemoryError('CVRootInit: Memory allocation error')

        jac = opts['jacfn']
        if jac is not None and not isinstance(jac , CV_JacRhsFunction):
            tmpfun = CV_WrapJacRhsFunction()
            tmpfun.set_jacfn(jac)
            jac = tmpfun
            opts['jacfn'] = tmpfun
        self.aux_data.jac = jac
        self.aux_data.user_data = opts['user_data']
        #self.aux_data.jac = opts['jacfn']
        self.aux_data.user_data = opts['user_data']

        self._set_runtime_changeable_options(opts, supress_supported_check=True)

        # As user data we pass the CV_data object
        CVodeSetUserData(cv_mem, <void*> self.aux_data)

        if (opts['order'] > 0):
            CVodeSetMaxOrd(cv_mem, <int> opts['order'])
        CVodeSetMaxNumSteps(cv_mem, <int> opts['max_steps'])
        if lmm_type == 'bdf':
            CVodeSetStabLimDet(cv_mem, <bint> opts['bdf_stability_detection'])
        if opts['first_step_size'] > 0.:
            CVodeSetInitStep(cv_mem, <realtype> opts['first_step_size'])
        if (opts['min_step_size'] > 0.):
           CVodeSetMinStep(cv_mem, <realtype> opts['min_step_size'])
        if opts['max_step_size'] > 0.:
            flag = CVodeSetMaxStep(cv_mem, <realtype> opts['max_step_size'])
            if flag == CV_ILL_INPUT:
                raise ValueError('CVodeSetMaxStep: max_step_size is negative '
                                 'or smaller than min_step_size.')
        if opts['max_nonlin_iters'] > 0:
            CVodeSetMaxNonlinIters(cv_mem, <int> opts['max_nonlin_iters'])
        if opts['max_conv_fails'] > 0:
            CVodeSetMaxConvFails(cv_mem, <int> opts['max_conv_fails'])
        if opts['nonlin_conv_coef'] > 0:
            CVodeSetNonlinConvCoef(cv_mem, <int> opts['nonlin_conv_coef'])

        # Linsolver
        if iter_type == 'newton':

            linsolver = opts['linsolver'].lower()

            if linsolver == 'dense':
                if self.parallel_implementation:
                    raise ValueError('Linear solver for dense matrices can be'
                                     'used only for serial implementation. For '
                                     'parallel implementation use ''lapackdense''.')
                else:
                    flag = CVDense(cv_mem, N)
                    if flag == CVDLS_ILL_INPUT:
                        raise ValueError('CVDense solver is not compatible with'
                                         ' the current nvector implementation.')
                    elif flag == CVDLS_MEM_FAIL:
                        raise MemoryError('CVDense memory allocation error.')
            elif linsolver == 'lapackdense':
                flag = CVLapackDense(cv_mem, N)
                if flag == CVDLS_ILL_INPUT:
                    raise ValueError('CVLapackDense solver is not compatible '
                                     'with the current nvector implementation.')
                elif flag == CVDLS_MEM_FAIL:
                    raise MemoryError('CVLapackDense memory allocation error.')
            elif linsolver == 'band':
                if self.parallel_implementation:
                    raise ValueError('Linear solver for band matrices can be '
                                     'used only for serial implementation. '
                                     'Use ''lapackband'' instead for parallel '
                                     'implementation.')
                else:
                    flag = CVBand(cv_mem, N, <int> opts['uband'],
                                             <int> opts['lband'])
                    if flag == CVDLS_ILL_INPUT:
                        raise ValueError('CVBand solver is not compatible with '
                                         'the current nvector implementation '
                                         'or bandwith outside range.')
                    elif flag == CVDLS_MEM_FAIL:
                        raise MemoryError('CVBand memory allocation error.')
            elif linsolver == 'lapackband':
                flag = CVLapackBand(cv_mem, N, <int> opts['uband'],
                                               <int> opts['lband'])
                if flag == CVDLS_ILL_INPUT:
                    raise ValueError('CVLapackBand solver is not compatible'
                                     ' with the current nvector implementation'
                                     ' or bandwith outside range.')
                elif flag == CVDLS_MEM_FAIL:
                    raise MemoryError('CVLapackBand memory allocation error.')
            elif linsolver == 'diag':
                flag = CVDiag(cv_mem)
                if flag == CVDIAG_ILL_INPUT:
                        raise ValueError('CVDiag solver is not compatible with'
                                         ' the current nvector implementation.')
                elif flag == CVDIAG_MEM_FAIL:
                        raise MemoryError('CVDiag memory allocation error.')
            elif ((linsolver == 'spgmr') or (linsolver == 'spbcg')
                  or (linsolver == 'sptfqmr')):
                precond_type = opts['precond_type'].lower()
                if precond_type == 'none':
                    pretype = PREC_NONE
                elif precond_type == 'left':
                    pretype = PREC_LEFT
                elif precond_type == 'right':
                    pretype = PREC_RIGHT
                elif precond_type == 'both':
                    pretype = PREC_BOTH
                else:
                    raise ValueError('LinSolver::Precondition: Unknown type: %s'
                                     % opts['precond_type'])

                if linsolver == 'spgmr':
                    flag = CVSpgmr(cv_mem, pretype, <int> opts['maxl'])
                elif linsolver == 'spbcg':
                    flag = CVSpbcg(cv_mem, pretype, <int> opts['maxl'])
                else:
                    flag = CVSptfqmr(cv_mem, pretype, <int> opts['maxl'])

                if flag == CVSPILS_MEM_FAIL:
                        raise MemoryError('LinSolver:CVSpils memory allocation '
                                          'error.')
            else:
                raise ValueError('LinSolver: Unknown solver type: %s'
                                     % opts['linsolver'])

        if (linsolver in ['dense', 'lapackdense']) and self.aux_data.jac:
            CVDlsSetDenseJacFn(cv_mem, _jacdense)
        #TODO: Reinitialization of the rooting function

        self.initialized = True

        return t0

    def solve(self, object tspan, object y0):

        cdef np.ndarray[DTYPE_t, ndim=1] np_tspan, np_y0
        np_tspan = np.asarray(tspan)
        np_y0    = np.asarray(y0)

        return self._solve(np_tspan, np_y0)

    cpdef _solve(self, np.ndarray[DTYPE_t, ndim=1] tspan,
                 np.ndarray[DTYPE_t, ndim=1] y0):

        cdef np.ndarray[DTYPE_t, ndim=1] t_retn
        cdef np.ndarray[DTYPE_t, ndim=2] y_retn
        t_retn  = np.empty(np.shape(tspan), float)
        y_retn  = np.empty([np.alen(tspan), np.alen(y0)], float)

        self._init_step(tspan[0], y0)
        t_retn[0] = tspan[0]
        y_retn[0, :] = y0

        cdef np.ndarray[DTYPE_t, ndim=1] y_last
        cdef unsigned int idx
        cdef DTYPE_t t
        cdef int flag
        cdef void *cv_mem = self._cv_mem
        cdef realtype t_out
        cdef N_Vector y  = self.y

        y_last   = np.empty(np.shape(y0), float)

        for idx in np.arange(np.alen(tspan))[1:]:
            t = tspan[idx]

            flag = CVode(cv_mem, <realtype> t,  y, &t_out, CV_NORMAL)

            nv_s2ndarray(y,  y_last)

            if flag != CV_SUCCESS:
                if flag == CV_TSTOP_RETURN:
                    print('Stop time reached... stopping computation...')
                elif flag == CV_ROOT_RETURN:
                    print('Found root... stopping computation...')
                elif flag < 0:
                    print('Error occured. See returned flag '
                          'variable and CVode documentation.')
                else:
                    print('Unhandled flag:', flag,
                          '\nComputation stopped... ')

                t_retn  = t_retn[0:idx]
                y_retn  = y_retn[0:idx, :]

                return flag, t_retn, y_retn, t_out, y_last

            t_retn[idx]    = t_out
            y_retn[idx, :] = y_last

        return flag, t_retn, y_retn, None, None

    def step(self, DTYPE_t t, np.ndarray[DTYPE_t, ndim=1] y_retn):
        """
        Method for calling successive next step of the CVODE solver to allow
        more precise control over the solver. The 'init_step' method has to
        be called before the 'step' method.

        Input:
            t - if t>0.0 then integration is performed until this time
                         and results at this time are returned in y_retn
              - if t<0.0 only one internal step is perfomed towards time abs(t)
                         and results after this one time step are returned
            y_retn - numpy vector (ndim = 1) in which the computed
                     value will be stored  (needs to be preallocated)
        Return values:
            flag  - status of the computation (successful or error occured)
            t_out - time, where the solver stopped
                    (when no error occured, t_out == t)
        """
        if not self.initialized:
            raise ValueError('Method ''init_step'' has to be called prior to '
                             'the first call of ''step'' method.')

        cdef N_Vector y  = self.y
        cdef realtype t_out
        cdef int flag

        if t>0.0:
            flag = CVode(self._cv_mem, <realtype> t,  y, &t_out, CV_NORMAL)
        else:
            flag = CVode(self._cv_mem, <realtype> t,  y, &t_out, CV_ONE_STEP)

        nv_s2ndarray(y, y_retn)

        return flag, t_out

    def __dealloc__(self):
        if not self._cv_mem is NULL: CVodeFree(&self._cv_mem)
        #TODO: when implementing parallel, does N_VDestroy be called separately
        #      for parallel version or it's a generic one?
        if not self.y0   is NULL: N_VDestroy(self.y0)
        if not self.y    is NULL: N_VDestroy(self.y)
        if not self.atol is NULL: N_VDestroy(self.atol)
