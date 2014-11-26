import inspect

import numpy as np
cimport numpy as np

from .c_sundials cimport realtype, N_Vector
from .c_ida cimport *
from .common_defs cimport (nv_s2ndarray, ndarray2nv_s, ndarray2DlsMatd)

# TODO: parallel implementation: N_VectorParallel
# TODO: linsolvers: check the output value for errors
# TODO: unify using float/double/realtype variable
# TODO: optimize code for compiler

# Right-hand side function
cdef class IDA_RhsFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       np.ndarray[DTYPE_t, ndim=1] result,
                       object userdata = None):
        return 0

cdef class IDA_WrapRhsFunction(IDA_RhsFunction):
    cpdef set_resfn(self, object resfn):
        """
        set some residual equations as a ResFunction executable class
        """
        self.with_userdata = 0
        nrarg = len(inspect.getargspec(resfn)[0])
        if nrarg > 5:
            # hopefully a class method
            self.with_userdata = 1
        elif nrarg == 5 and inspect.isfunction(resfn):
            self.with_userdata = 1
        self._resfn = resfn

    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       np.ndarray[DTYPE_t, ndim=1] result,
                       object userdata = None):
        if self.with_userdata == 1:
            self._resfn(t, y, ydot, result, userdata)
        else:
            self._resfn(t, y, ydot, result)
        return 0

cdef int _res(realtype tt, N_Vector yy, N_Vector yp,
              N_Vector rr, void *auxiliary_data):
    """ function with the signature of IDAResFn """
    cdef np.ndarray[DTYPE_t, ndim=1] residual_tmp, yy_tmp, yp_tmp

    aux_data = <IDA_data> auxiliary_data
    cdef bint parallel_implementation = aux_data.parallel_implementation

    if parallel_implementation:
        raise NotImplemented
    else:
        yy_tmp = aux_data.yy_tmp
        yp_tmp = aux_data.yp_tmp
        residual_tmp = aux_data.residual_tmp

        nv_s2ndarray(yy, yy_tmp)
        nv_s2ndarray(yp, yp_tmp)

    aux_data.res.evaluate(tt, yy_tmp, yp_tmp, residual_tmp, aux_data.user_data)

    if parallel_implementation:
        raise NotImplemented
    else:
        ndarray2nv_s(rr, residual_tmp)

    return 0

# Root function
cdef class IDA_RootFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       np.ndarray[DTYPE_t, ndim=1] g,
                       object userdata = None):
        return 0

cdef class IDA_WrapRootFunction(IDA_RootFunction):
    cpdef set_rootfn(self, object rootfn):
        """
        set root-ing condition(equations) as a RootFunction executable class
        """
        self.with_userdata = 0
        nrarg = len(inspect.getargspec(rootfn)[0])
        if nrarg > 5:
            #hopefully a class method, self gives 5 arg!
            self.with_userdata = 1
        elif nrarg == 5 and inspect.isfunction(rootfn):
            self.with_userdata = 1
        self._rootfn = rootfn

    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       np.ndarray[DTYPE_t, ndim=1] g,
                       object userdata = None):
        if self.with_userdata == 1:
            self._rootfn(t, y, ydot, g, userdata)
        else:
            self._rootfn(t, y, ydot, g)
        return 0

cdef int _rootfn(realtype t, N_Vector yy, N_Vector yp,
                 realtype *gout, void *auxiliary_data):
    """ function with the signature of IDARootFn """

    aux_data = <IDA_data> auxiliary_data
    cdef bint parallel_implementation = aux_data.parallel_implementation

    if parallel_implementation:
        raise NotImplemented
    else:
        yy_tmp = aux_data.yy_tmp
        yp_tmp = aux_data.yp_tmp
        g_tmp  = aux_data.g_tmp

        nv_s2ndarray(yy, yy_tmp)
        nv_s2ndarray(yp, yp_tmp)

    aux_data.rootfn.evaluate(t, yy_tmp, yp_tmp, g_tmp, aux_data.user_data)

    cdef int i
    if parallel_implementation:
        raise NotImplemented
    else:
        for i in np.arange(np.alen(g_tmp)):
            gout[i] = <realtype> g_tmp[i]

    return 0

# Jacobian function
cdef class IDA_JacRhsFunction:
    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       DTYPE_t cj,
                       np.ndarray J):
        """
        Returns the Jacobi matrix of the residual function, as
            d(res)/d y + cj d(res)/d ydot
        (for dense the full matrix, for band only bands). Result has to be
        stored in the variable J, which is preallocated to the corresponding
        size.

        This is a generic class, you should subclass is for the problem specific
        purposes."
        """
        return 0

cdef class IDA_WrapJacRhsFunction(IDA_JacRhsFunction):
    cpdef set_jacfn(self, object jacfn):
        """
        Set some jacobian equations as a JacResFunction executable class.
        """
        self._jacfn = jacfn

    cpdef int evaluate(self, DTYPE_t t,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       np.ndarray[DTYPE_t, ndim=1] ydot,
                       DTYPE_t cj,
                       np.ndarray J):
        """
        Returns the Jacobi matrix (for dense the full matrix, for band only
        bands. Result has to be stored in the variable J, which is preallocated
        to the corresponding size.
        """
##        if self.with_userdata == 1:
##            self._jacfn(t, y, ydot, cj, J, userdata)
##        else:
##            self._jacfn(t, y, ydot, cj, J)
        self._jacfn(t, y, ydot, cj, J)
        return 0

cdef int _jacdense(long int Neq, realtype tt, realtype cj,
            N_Vector yy, N_Vector yp, N_Vector rr, DlsMat Jac,
            void *auxiliary_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3):
    """function with the signature of IDADlsDenseJacFn """
    cdef np.ndarray[DTYPE_t, ndim=1] yy_tmp, yp_tmp
    cdef np.ndarray jac_tmp

    aux_data = <IDA_data> auxiliary_data
    cdef bint parallel_implementation = aux_data.parallel_implementation
    if parallel_implementation:
        raise NotImplemented
    else:
        yy_tmp = aux_data.yy_tmp
        yp_tmp = aux_data.yp_tmp
        if aux_data.jac_tmp == None:
            N = np.alen(yy_tmp)
            aux_data.jac_tmp = np.empty((N,N), float)
        jac_tmp = aux_data.jac_tmp

        nv_s2ndarray(yy, yy_tmp)
        nv_s2ndarray(yp, yp_tmp)
    aux_data.jac.evaluate(tt, yy_tmp, yp_tmp, cj, jac_tmp)

    if parallel_implementation:
        raise NotImplemented
    else:
        #we convert the python jac_tmp array to DslMat of sundials
        ndarray2DlsMatd(Jac, jac_tmp)

    return 0

cdef class IDA_data:
    def __cinit__(self, N):
        self.parallel_implementation = False
        self.user_data = None

        self.yy_tmp = np.empty(N, float)
        self.yp_tmp = np.empty(N, float)
        self.residual_tmp = np.empty(N, float)
        self.jac_tmp = None
        self.g_tmp = None

cdef class IDA:

    def __cinit__(self, Rfn, **options):
        """
        Initialize the IDA Solver and it's default values

        Input:
            Rfn     - residual function
            options - additional options for initialization
        """

        default_values = {
            'verbosity': 1,
            'implementation': 'serial',
            'rtol': 1e-6, 'atol': 1e-12,
            'linsolver': 'dense',
            'lband': 0,'uband': 0,
            'maxl': 0,
            'tstop': 0.,
            'order': 0,
            'max_step_size': 0.,
            'first_step_size': 0.,
            'max_steps': 0,
            'max_conv_fails': 0,  # sundials default is 10
            'max_nonlin_iters': 0,
            'nonlin_conv_coef': 0.,
            'compute_initcond': None,
            'compute_initcond_t0': 0.01,
            'constraints_idx': None,
            'constraints_type': None,
            'algebraic_vars_idx':None,
            'exclude_algvar_from_error': False,
            'user_data': None,
            'rfn': None,
            'rootfn': None,
            'nr_rootfns': 0,
            'jacfn': None
            }

        self.verbosity = 1
        self.options = default_values
        self.N       = -1
        self.set_options(rfn=Rfn, **options)
        self._ida_mem = NULL
        self.initialized = False

    def set_options(self, **options):
        """
        Reads the options list and assigns values for the solver.

        All options list:
            'verbosity':
                Values: 0,1,2,...
                Description:
                    Set the level of verbosity. The higher number user, the
                    more verbose the output will be. Default is 1.
            'implementation':
                Values: 'serial' (= default), 'parallel'
                Description:
                    Using serial or parallel implementation of the solver.
                    #TODO: curently only serial implementation is working
            'use_relaxation':
                Values: False (= default), True
                Description:
                    Uses relaxation algorithm for solving (if possible).
            'rtol':
                Values: float,  1e-6 = default
                Description:
                    Relative tolerancy.
            'atol':
                Values: float or numpy array of floats,  1e-12 = default
                Description:
                    Absolute tolerancy
            'linsolver':
                Values: 'dense' (= default), 'lapackdense', 'band',
                        'lapackband', 'spgmr', 'spbcg', 'sptfqmr'
                Description:
                    Specifies used linear solver.
                    Limitations: Linear solvers for dense and band matrices can
                        be used only for serial implementation. For parallel
                        implementation use lapackdense or lapackband
                         respectively.
            'lband', 'uband':
                Values: non-negative integer, 0 = default
                Description:
                    Specifies the width of the lower band (below the main
                    diagonal) and/or upper diagonal (above the main diagonal).
                    So the total band width is lband + 1 + uband, where the 1
                    stands for the main diagonal (specially, lband = uband = 0
                    [that is the default] denotes the band width = 1, i.e. the
                    main diagonal only). Used only if 'linsolver' is band.
            'maxl':
                Values: 0 (= default), 1, 2, 3, 4, 5
                Description:
                    Dimension of the number of used Krylov subspaces
                    (used only by 'spgmr', 'spbcg', 'sptfqmr' linsolvers)
            'tstop':
                Values: float, 0.0 = default
                Description:
                    Maximum time until which we perform the computations.
                    Default is 0.0. Once the value is set 0.0, it cannot be
                    disable (but it will be automatically disable when tstop
                    is reached and has to be reset again in next run).
            'order':
                Values: 1, 2, 3, 4, 5 (= default)
                Description:
                    Specifies the maximum order of the linear multistep method.
            'max_steps':
                Values: positive integer, 0 = default (uses value of 500)
                Description:
                    Maximum number of (internally defined) steps allowed during
                    one call to the solver.
            'max_step_size':
                Values: non-negative float, 0.0 = default
                Description:
                    Restricts the maximal (absolute) internal step value taken
                    by the solver. The value of 0.0 uses no restriction on the
                    maximal step size.
            'first_step_size':
                Values: float , 0.0 = default
                Description:
                    Sets the first step size. DAE solver can suffer on the
                    first step, so set this to circumvent this. The value of 0.0
                    uses the solver's internal default value.
            'compute_initcond':
                Values: 'y0', 'yp0', None
                Description:
                    Specifies what initial condition is calculated for given
                    problem.

                    'y0'  - calculates/improves the y0 initial condition
                            (considering yp0 to be accurate)
                    'yp0' - calculates/improves the yp0 initial condition
                            (considering y0 to be accurate)
                    None  - don't compute any initial conditions
                            (user provided values for y0 an yp0 are considered
                             to be consistent and accurate)
            'compute_initcond_t0':
                Values: non-negative float, 0.01 = default
                Description:
                    When calculating the initial condition, specifies the time
                    until which the solver tries to
                    get the consistent values for either y0 or yp0.
            'user_data':
                Values: python object, None = default
                Description:
                    Additional data that are supplied to each call of the
                    residual function 'Rfn' (see below), root function 'rootfn'
                    and Jacobi function 'jacfn' (if specified one).
            'Rfn':
                Values: function of class IDA_RhsFunction or a python function
                    with a signature (t, y, yp, resultout)
                Description:
                    Defines the residual function (which has to be a subclass of
                    IDA_RhsFunction class, or a normal python function with
                    signature (t, y, yp, resultout) ).
                    This function takes as input arguments current time t,
                    current value of y, yp, numpy array of returned residual
                    and optional userdata. Return value is 0 if successfull.
                    This option is mandatory.
            'rootfn':
                Values: function of class IDA_RootFunction or a python function
                    with signature (t, y, yp, g, user_data)
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
                Values: function of class IDA_JacRhFunction or python function
                Description:
                    Defines the jacobian of the residual function as
                                   dres/dy + cj dres/dyp,
                    and has to be a subclass of IDA_JacRhFunction class or
                    a python function.
                    This function takes as input arguments current time t,
                    current value of y, yp, cj, and 2D numpy array of returned
                    jacobian
                    and optional userdata. Return value is 0 if successfull.
                    Jacobian is only used for dense or lapackdense linear solver
            'algebraic_vars_idx':
                Values: numpy vector or None (= default)
                Description:
                    If given problem is of type DAE, some items of the residual
                    vector returned by the 'resfn' have to be treated as
                    algebraic variables. These are denoted by the position
                    (index) in the residual vector.
                    All these indexes have to be specified in the
                    'algebraic_vars_idx' array.
            'exclude_algvar_from_error':
                Values: False (= default), True
                Description:
                    Indicates whether or not to suppress algebraic variables in
                    the local error test. If 'algebraic_vars_idx'
                    vector not specified, this value is ignored.

                    The use of this option (i.e. set to True) is discouraged
                    when solving DAE systems of index 1, whereas it
                    is generally encouraged for systems of index 2 or more.
            'constraints_idx':
                Values: numpy vector or None (= default)
                Description:
                    Constraint the individual variables. The variables are
                    denoted by the position (index) in the residual vector.
                    All these indexes are/have to be specified in this
                    'constraints_idx' array.
            'constraints_type':
                Values: numpy vector or None (= default)
                Description:
                    The actuall type of contraints applied to the of the
                    variables specified by 'constraints_idx'. The type is
                    one of:  0.0 - no contraint
                             1.0 - variable has to be non-negative (i.e. >= 0)
                             2.0 - variable has to be positive (i.e. > 0)
                            -1.0 - variable has to be non-positive (i.e. <= 0)
                            -2.0 - variable has to be negative (i.e. < 0)
        """

        for (key, value) in options.items():
            if key.lower() in self.options:
                self.options[key.lower()] = value
            else:
                raise ValueError("Option '%s' is not supported by solver" % key)

        self._set_runtime_changeable_options(options)

    cpdef _set_runtime_changeable_options(self, object options,
                                          bint supress_supported_check=False):
        """
          (Re)Set options that can change after IDA_MEM object was allocated -
          mostly useful for values that need to be changed at run-time.
        """
        cdef int flag
        cdef void* ida_mem = self._ida_mem

        if ida_mem is NULL:
            return 0

        # Check whether options are all only supported - this check can
        # be supressed by 'supress_supported_check = True'
        if not supress_supported_check:
            for opt in options.keys():
                if not opt in ['atol', 'rtol', 'tstop', 'rootfn', 'nr_rootfns']:
                    raise ValueError("Option '%s' can''t be set runtime." % opt)

        # Verbosity level
        if ('verbosity' in options) and (options['verbosity'] is not None):
            verbosity = options['verbosity']
            self.options['verbosity'] = verbosity
            self.verbosity = verbosity

        # Root function
        if ('rootfn' in options) and (options['rootfn'] is not None):
            # TODO: Unsetting the rootfn?
            rootfn = options['rootfn']
            self.options['rootfn'] = rootfn

            nr_rootfns = options['nr_rootfns']
            self.options['nr_rootfns'] = nr_rootfns
            if nr_rootfns is None:
                raise ValueError('Number of root-ing functions ''nr_rootfns'' '
                                 'must be specified.')
            if not isinstance(rootfn, IDA_RootFunction):
                tmpfun = IDA_WrapRootFunction()
                tmpfun.set_rootfn(rootfn)
                rootfn = tmpfun
                self.options['rootfn'] = tmpfun

            self.aux_data.rootfn = rootfn
            self.aux_data.g_tmp  = np.empty([nr_rootfns,], float)

            flag = IDARootInit(ida_mem, nr_rootfns, _rootfn)

            if flag == IDA_SUCCESS:
                pass
            if flag == IDA_ILL_INPUT:
                raise ValueError('IDARootInit: Function ''rootfn'' is NULL '
                                 'but ''nr_rootfns'' > 0')
            elif flag == IDA_MEM_FAIL:
                raise MemoryError('IDARootInit: Memory allocation error')


        # Set tolerances
        cdef N_Vector atol
        cdef np.ndarray[DTYPE_t, ndim=1] np_atol

        if 'atol' in options:
            opts_atol = options['atol']
            self.options['atol'] = opts_atol
        else:
            opts_atol = None

        if 'rtol' in options:
            opts_rtol = options['rtol']
            self.options['rtol'] = opts_rtol
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
                flag = IDASStolerances(ida_mem, <realtype> opts_rtol,
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
                    flag = IDASVtolerances(ida_mem, <realtype> opts_rtol, atol)

                    self.atol = atol

            if flag == IDA_ILL_INPUT:
                raise ValueError("IDATolerances: negative 'atol' or 'rtol' value.")
        #TODO: implement IDAFWtolerances(ida_mem, efun)

        # Set tstop
        if ('tstop' in options) and (options['tstop'] is not None):
            opts_tstop = options['tstop']
            self.options['tstop'] = opts_tstop
            if (not opts_tstop is None) and (opts_tstop > 0.):
               flag = IDASetStopTime(ida_mem, <realtype> opts_tstop)
               if flag == IDA_ILL_INPUT:
                   raise ValueError('IDASetStopTime::Stop value is beyond '
                                    'current value.')

    def init_step(self, double t0, object y0, object yp0,
                   np.ndarray y_ic0_retn = None,
                   np.ndarray yp_ic0_retn = None):
        """
        Initializes the solver and allocates memory.

        Input:
            t0     - initial time
            y0     - initial condition for y (can be list or numpy array)
            yp0    - initial condition for yp (can be list or numpy array)
            y_ic0  - (optional) returns the calculated consistent initial
                     condition for y
                     It MUST be a numpy array.
            yp_ic0 - (optional) returns the calculated consistent initial
                     condition for y derivated. It MUST be a numpy array.
        """

        cdef np.ndarray[DTYPE_t, ndim=1] np_y0
        cdef np.ndarray[DTYPE_t, ndim=1] np_yp0
        np_y0  = np.asarray(y0, dtype=float)
        np_yp0 = np.asarray(yp0, dtype=float)

        return self._init_step(t0, np_y0, np_yp0, y_ic0_retn, yp_ic0_retn)

    cpdef _init_step(self, DTYPE_t t0,
                     np.ndarray[DTYPE_t, ndim=1] y0,
                     np.ndarray[DTYPE_t, ndim=1] yp0,
                     np.ndarray y_ic0_retn = None,
                     np.ndarray yp_ic0_retn = None):
        """
            Applies the set by 'set_options' method to the IDA solver.

            Performs computation of initial conditions (if compute_initcond
            flag set). Used only in conjuction with the 'step' method to provide
            initialization and assure that the initial condition is calculated.

            Input:
                y0 - initial condition for y
                yp0 - initial condition for derivation of y
                y_ic0_retn  - (optional) returns the calculated consistent
                              initial condition for y
                yp_ic0_retn - (optional) returns the calculated consistent
                              initial condition for y derivated

            Note: After setting (or changing) options with 'set_options' method
                  you need to call 'init_step' prior to the 'step' method to
                  assure the values are correctly initialized.

            Returns the time of the intial step.
        """
        #TODO: jacobi function ?isset

        cdef dict opts = self.options

        self.parallel_implementation = (opts['implementation'].lower() == 'parallel')
        if self.parallel_implementation:
            raise ValueError('Error: Parallel implementation not implemented !')
        cdef long int N
        N = <long int> np.alen(y0)

        if opts['rfn'] == None:
            raise ValueError('The residual function rfn not assigned '
                              'during ''set_options'' call !')

        if not np.alen(y0) == np.alen(yp0):
            raise ValueError('Arrays inconsistency: y0 and ydot0 have to be of '
                             'the same length !')
        if (((np.alen(y0) == 0) and (not opts['compute_initcond'] == 'y0'))
                or ((np.alen(yp0) == 0) and (not opts['compute_initcond'] == 'yp0'))):
            raise ValueError('Value of y0 not set or the value of ydot0 not '
                             'flagged to be computed (see ''init_cond'' for '
                             'documentation.')

        #TODO: when implementing parallel, does N_VDestroy be called separately
        #      for parallel version or it's a generic one?
        if not self.y0 is NULL:
            N_VDestroy(self.y0)
            N_VDestroy(self.yp0)
            N_VDestroy(self.residual)
            N_VDestroy(self.y)
            N_VDestroy(self.yp)

        if self.parallel_implementation:
            raise NotImplemented
        else:
            self.y0  = N_VMake_Serial(N, <realtype *>y0.data)
            self.yp0 = N_VMake_Serial(N, <realtype *>yp0.data)
            self.residual = N_VNew_Serial(N)
            self.y   = N_VClone(self.y0)
            self.yp  = N_VClone(self.yp0)

        cdef int flag
        cdef void* ida_mem = self._ida_mem

        if (ida_mem is NULL) or (self.N != N):
            if (not ida_mem is NULL):
                IDAFree(&ida_mem)
            ida_mem = IDACreate()
            if ida_mem is NULL:
                raise MemoryError('IDACreate:MemoryError: Could not allocate '
                                  'ida_mem object')

            self._ida_mem = ida_mem
            flag = IDAInit(ida_mem, _res, <realtype> t0, self.y0, self.yp0)
        elif self.N == N:
            flag = IDAReInit(ida_mem, <realtype> t0, self.y0, self.yp0)
        else:
            raise ValueError('IDAInit:Error: You should not be here...')
        if flag == IDA_ILL_INPUT:
                raise ValueError('IDA[Re]Init: Ill input')
        elif flag == IDA_MEM_FAIL:
            raise MemoryError('IDA[Re]Init: Memory allocation error')
        elif flag == IDA_MEM_NULL:
            raise MemoryError('IDACreate: Memory allocation error')
        elif flag == IDA_NO_MALLOC:
            raise MemoryError('IDAReInit: No memory allocated in IDAInit.')

        self.N = N

        # auxiliary variables
        self.aux_data = IDA_data(N)
        self.aux_data.parallel_implementation = self.parallel_implementation

        rfn = opts['rfn']
        if not isinstance(rfn , IDA_RhsFunction):
            tmpfun = IDA_WrapRhsFunction()
            tmpfun.set_resfn(rfn)
            rfn = tmpfun
            opts['rfn'] = tmpfun
        self.aux_data.res = rfn

        jac = opts['jacfn']
        if jac is not None and not isinstance(jac , IDA_JacRhsFunction):
            tmpfun = IDA_WrapJacRhsFunction()
            tmpfun.set_jacfn(jac)
            jac = tmpfun
            opts['jacfn'] = tmpfun
        self.aux_data.jac = jac
        self.aux_data.user_data = opts['user_data']

        self._set_runtime_changeable_options(opts, supress_supported_check=True)

        if flag == IDA_ILL_INPUT:
            raise ValueError('IDAStolerances: negative atol or rtol value.')

        # As user data we pass the IDA_data object
        IDASetUserData(ida_mem, <void*> self.aux_data)

        if (opts['order'] > 0):
            IDASetMaxOrd(ida_mem, <int> opts['order'])
        IDASetMaxNumSteps(ida_mem, <int> opts['max_steps'])
        if opts['first_step_size'] > 0.:
            IDASetInitStep(ida_mem, <realtype> opts['first_step_size'])
        if opts['max_step_size'] > 0.:
            flag = IDASetMaxStep(ida_mem, <realtype> opts['max_step_size'])
            if flag == IDA_ILL_INPUT:
                raise ValueError('IDASetMaxStep: max_step_size is negative or '
                                 'smaller than allowed.')
        if opts['max_nonlin_iters'] > 0:
            IDASetMaxNonlinIters(ida_mem, <int> opts['max_nonlin_iters'])
        if opts['max_conv_fails'] > 0:
            IDASetMaxConvFails(ida_mem, <int> opts['max_conv_fails'])
        if opts['nonlin_conv_coef'] > 0:
            IDASetNonlinConvCoef(ida_mem, <int> opts['nonlin_conv_coef'])

        # Linsolver
        linsolver = opts['linsolver'].lower()

        if linsolver == 'dense':
            if self.parallel_implementation:
                raise ValueError('Linear solver for dense matrices can be used '
                                  'only for serial implementation. For parallel'
                                  ' implementation use ''lapackdense'' instead.')
            else:
                 flag = IDADense(ida_mem, N)
                 if flag == IDADLS_ILL_INPUT:
                     raise ValueError('IDADense solver is not compatible with'
                                      ' the current nvector implementation.')
                 elif flag == IDADLS_MEM_FAIL:
                     raise MemoryError('IDADense memory allocation error.')
        elif linsolver == 'lapackdense':
            flag = IDALapackDense(ida_mem, N)
            if flag == IDADLS_ILL_INPUT:
                raise ValueError('IDALapackDense solver is not compatible with'
                                 ' the current nvector implementation.')
            elif flag == IDADLS_MEM_FAIL:
                raise MemoryError('IDALapackDense memory allocation error.')
        elif linsolver == 'band':
            if self.parallel_implementation:
                raise ValueError('Linear solver for band matrices can be used'
                                 'only for serial implementation. For parallel'
                                 ' implementation use ''lapackband'' instead.')
            else:
                flag = IDABand(ida_mem, N, <int> opts['uband'],
                                           <int> opts['lband'])
                if flag == IDADLS_ILL_INPUT:
                    raise ValueError('IDABand solver is not compatible'
                                     ' with the current nvector implementation'
                                     ' or bandwith outside range.')
                elif flag == IDADLS_MEM_FAIL:
                    raise MemoryError('IDABand memory allocation error.')
        elif linsolver == 'lapackband':
            flag = IDALapackBand(ida_mem, N, <int> opts['uband'],
                                             <int> opts['lband'])
            if flag == IDADLS_ILL_INPUT:
                raise ValueError('IDALapackBand solver is not compatible'
                                 ' with the current nvector implementation'
                                 ' or bandwith outside range.')
            elif flag == IDADLS_MEM_FAIL:
                raise MemoryError('IDALapackBand memory allocation error.')
        elif ((linsolver == 'spgmr') or (linsolver == 'spbcg')
                  or (linsolver == 'sptfqmr')):
            maxl = <int> opts['maxl']

            if linsolver == 'spgmr':
                flag = IDASpgmr(ida_mem, maxl)
            elif linsolver == 'spbcg':
                flag = IDASpbcg(ida_mem, maxl)
            else:
                flag = IDASptfqmr(ida_mem, maxl)

            if flag == IDASPILS_MEM_FAIL:
                raise MemoryError('LinSolver:IDASpils memory allocation error.')

        if (linsolver in ['dense', 'lapackdense']) and self.aux_data.jac:
            IDADlsSetDenseJacFn(ida_mem, _jacdense)

        # Constraints
        constraints_idx  = opts['constraints_idx']
        constraints_type = opts['constraints_type']
        cdef unsigned int idx
        cdef np.ndarray[DTYPE_t, ndim=1] constraints_vars

        if not constraints_type is None:
            if not constraints_idx is None:
                constraints_vars = np.zeros(N, float)
                for idx in range(constraints_idx):
                    constraints_vars[constraints_idx[idx]] = constraints_type[idx]
            else:
                assert np.alen(constraints_type) == N,\
                  'Without ''constraints_idx'' specified the'\
                  '  ''constraints_type'' the constraints_idx has to be of the'\
                  ' same length as y (i.e. contains constraints for EVERY '\
                  'component of y).'
                constraints_vars = np.asarray(constraints_type, float)

            if not self.constraints is NULL:
                N_VDestroy(self.constraints)
                self.constraints = NULL

            if self.parallel_implementation:
                raise NotImplemented
            else:
                self.constraints =\
                   N_VMake_Serial(N, <realtype*> constraints_vars.data)
            flag = IDASetConstraints(ida_mem, self.constraints)
            if flag == IDA_ILL_INPUT:
                raise ValueError('IDAConstraints: ''constraints_type'' contains'
                                 'illegal value.')

        # DAE variables and Initial condition
        cdef np.ndarray[DTYPE_t, ndim=1] dae_vars

        alg_vars_idx     = opts['algebraic_vars_idx']

        if opts['compute_initcond']:
            compute_initcond = opts['compute_initcond'].lower()
        else:
            compute_initcond = ''

        if ((compute_initcond == 'yp0') or
            (opts['exclude_algvar_from_error'] and not alg_vars_idx is None)):

            # DAE variables
            dae_vars = np.ones(N, float)

            if not alg_vars_idx is None:
                for algvar_idx in opts['algebraic_vars_idx']:
                    dae_vars[algvar_idx] = 0.0

            if not self.dae_vars_id is NULL:
                N_VDestroy(self.dae_vars_id)
                self.dae_vars_id = NULL

            if self.parallel_implementation:
                raise NotImplemented
            else:
                self.dae_vars_id = N_VMake_Serial(N, <realtype*> dae_vars.data)

            IDASetId(ida_mem, self.dae_vars_id)

            if opts['exclude_algvar_from_error'] and not alg_vars_idx is None:
                IDASetSuppressAlg(ida_mem, True)

        # Initial condition
        cdef bint compute_initcond_p
        cdef float t0_init
        cdef realtype ic_t0 = <realtype>opts['compute_initcond_t0']

        if compute_initcond in [None, 'y0', 'yp0', '']:
            compute_initcond_p = (compute_initcond == 'y0' or
                                  compute_initcond == 'yp0')
        else:
            raise ValueError('InitCond: Unknown ''compute_initcond'' value: %s'
                             % compute_initcond)

        if compute_initcond_p:
            if compute_initcond == 'yp0':
                flag = IDACalcIC(ida_mem, IDA_YA_YDP_INIT, ic_t0)
            elif compute_initcond == 'y0':
                flag = IDACalcIC(ida_mem, IDA_Y_INIT, ic_t0)

            if not (flag == IDA_SUCCESS):
                print('IDAInitCond: Error occured during computation'
                      ' of initial condition, flag', flag)
                return (False, t0)

            t0_init = ic_t0
        else:
            t0_init = t0

        if not ((y_ic0_retn is None) and (yp_ic0_retn is None)):
            assert np.alen(y_ic0_retn) == np.alen(yp_ic0_retn) == N,\
              'y_ic0 and/or yp_ic0 have to be of the same size as y0.'

            flag = IDAGetConsistentIC(self._ida_mem, self.y, self.yp)

            if flag == IDA_ILL_INPUT:
                raise NameError('Method ''get_consistent_ic'' has to be called'
                                ' prior the first call to the ''step'' method.')

            if not y_ic0_retn is None: nv_s2ndarray(self.y, y_ic0_retn)
            if not yp_ic0_retn is None: nv_s2ndarray(self.yp, yp_ic0_retn)

        #TODO: implement the rest of IDA*Set* functions for linear solvers

        #TODO: Reinitialization of the rooting function

        self.initialized = True

        return (True, t0_init)

    def solve(self, object tspan, object y0,  object yp0):
        """
        Runs the solver.

        Input:
            tspan - an numpy array of times at which the computed value will be
                    returned
            y0    - numpy array of initial values
            yp0   - numpy array of initial values of derivatives

        Return values:
            flag   - indicating return status of the solver
            t      - numpy array of times at which the computations were
                     successful
            y      - numpy array of values corresponding to times t
                     (values of y[i, :] ~ t[i])
            yp     - numpy array of derivatives corresponding to times t
                     (values of yp[i, :] ~ t[i])
            t_err  - float or None - if recoverable error occured (for example,
                     reached the maximum number of allowed iterations); this is
                     the time at which it happened
            y_err  - numpy array of values corresponding to time t_err
            yp_err - numpy array of derivatives corresponding to time t_err

        Note:
            If 'calc_initcond' then solver returns instead of user supplied
            y0, yp0 values as the starting values the values calculated by the
            solver (i.e. consistent initial conditions. The starting time is
            then also the precomputed time.
        """

        cdef np.ndarray[DTYPE_t, ndim=1] np_tspan, np_y0, np_yp0

        np_tspan = np.asarray(tspan)
        np_y0    = np.asarray(y0)
        np_yp0   = np.asarray(yp0)

        return self._solve(np_tspan, np_y0, np_yp0)

    cpdef _solve(self, np.ndarray[DTYPE_t, ndim=1] tspan,
                       np.ndarray[DTYPE_t, ndim=1] y0,
                       np.ndarray[DTYPE_t, ndim=1] yp0):


        cdef np.ndarray[DTYPE_t, ndim=1] t_retn
        cdef np.ndarray[DTYPE_t, ndim=2] y_retn, yp_retn
        t_retn  = np.empty([np.alen(tspan), ], float)
        y_retn  = np.empty([np.alen(tspan), np.alen(y0)], float)
        yp_retn = np.empty([np.alen(tspan), np.alen(y0)], float)

        cdef int flag

        (flag, t_retn[0]) = self.init_step(tspan[0], y0, yp0,
                                           y_retn[0, :], yp_retn[0, :])
        if not flag:
            return (False, t_retn[0], y0, None, None, None, None)
        #TODO: Parallel version
        cdef np.ndarray[DTYPE_t, ndim=1] y_last, yp_last
        cdef unsigned int idx
        cdef DTYPE_t t
        cdef void *ida_mem = self._ida_mem
        cdef realtype t_out
        cdef N_Vector y  = self.y
        cdef N_Vector yp = self.yp

        y_last   = np.empty(np.shape(y0), float)
        yp_last  = np.empty(np.shape(y0), float)

        for idx in np.arange(np.alen(tspan))[1:]:
            t = tspan[idx]

            flag = IDASolve(self._ida_mem, <realtype> t, &t_out, y, yp,
                            IDA_NORMAL)

            nv_s2ndarray(y,  y_last)
            nv_s2ndarray(yp,  yp_last)

            if flag != IDA_SUCCESS:
                if flag == IDA_TSTOP_RETURN:
                    if self.verbosity > 1:
                        print('Stop time reached... stopping computation...')
                elif flag == IDA_ROOT_RETURN:
                    if self.verbosity > 1:
                        print('Found root... stopping computation...')
                elif flag < 0:
                    print('Error occured. See returned flag '
                          'variable and IDA documentation.')
                else:
                    print('Unhandled flag:', flag,
                          '\nComputation stopped... ')

                t_retn   = t_retn[0:idx]
                y_retn   = y_retn[0:idx, :]
                yp_retn  = yp_retn[0:idx, :]

                return flag, t_retn, y_retn, yp_retn, t_out, y_last, yp_last

            t_retn[idx]     = t_out
            y_retn[idx, :]  = y_last
            yp_retn[idx, :] = yp_last

        return flag, t_retn, y_retn, yp_retn, None, None, None

    def step(self, DTYPE_t t, np.ndarray[DTYPE_t, ndim=1] y_retn = None,
                              np.ndarray[DTYPE_t, ndim=1] yp_retn = None):
        """
        Method for calling successive next step of the IDA solver to allow
        more precise control over the IDA solver. The 'init_step' method has to
        be called before the 'step' method.

        Input:
            t - if t>0.0 then integration is performed until this time
                         and results at this time are returned in y_retn
              - if t<0.0 only one internal step is perfomed towards time abs(t)
                         and results after this one time step are returned
            y_retn - numpy vector (ndim = 1) in which the computed
                     value will be stored  (needs to be preallocated)
            yp_retn - numpy vector (ndim = 1) or None. If not None, will be
                      filled (needs to be preallocated)
                      with derivatives of y at time t.
        Return values:
            flag  - status of the computation (successful or error occured)
            t_out - time, where the solver stopped (when no error occured,
                    t_out == t)
        """
        if not self.initialized:
            raise ValueError('Method ''init_step'' has to be called prior to'
                             'the first call of ''step'' method.')
        cdef N_Vector y  = self.y
        cdef N_Vector yp = self.yp
        cdef realtype t_out
        cdef int flag
        if t>0.0:
            flag = IDASolve(self._ida_mem, <realtype> t, &t_out, y, yp,
                            IDA_NORMAL)
        else:
            flag = IDASolve(self._ida_mem, <realtype> -t, &t_out, y, yp,
                            IDA_ONE_STEP)
        if not y_retn is None:
            nv_s2ndarray(y, y_retn)
        if not yp_retn is None:
            nv_s2ndarray(yp, yp_retn)

        return flag, t_out

    def __dealloc__(self):
        if not self._ida_mem is NULL: IDAFree(&self._ida_mem)
        #TODO: when implementing parallel, does N_VDestroy be called separately
        #      for parallel version or it's a generic one?
        if not self.atol is NULL: N_VDestroy(self.atol)
        if not self.y0 is NULL: N_VDestroy(self.y0)
        if not self.yp0 is NULL: N_VDestroy(self.yp0)
        if not self.y is NULL: N_VDestroy(self.y)
        if not self.yp is NULL: N_VDestroy(self.yp)
        if not self.residual is NULL: N_VDestroy(self.residual)
        if not self.dae_vars_id is NULL: N_VDestroy(self.dae_vars_id)
        if not self.constraints is NULL: N_VDestroy(self.constraints)
