import numpy as np
cimport numpy as np

from c_sundials cimport realtype, N_Vector
from c_ida cimport *
from common_defs cimport *

# TODO: parallel implementation: N_VectorParallel
# TODO: linsolvers: check the output value for errors
# TODO: flag for indicating the resfn (in set_options) whether is a c-function or python function
# TODO: implement event handler
# TODO: unify using float/double/realtype variable
# TODO: optimize code for compiler

cdef int _res(realtype tt, N_Vector yy, N_Vector yp,
              N_Vector rr, void *auxiliary_data):

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
    
cdef class IDA_data:
    def __cinit__(self, N):
        self.parallel_implementation = False
        self.user_data = None
        
        self.yy_tmp = np.empty(N, float)
        self.yp_tmp = np.empty(N, float)
        self.residual_tmp = np.empty(N, float)
    
cdef class IDA:
    cdef realtype _jacFn(self, Neq, tt, yy, yp, resvec, cj, jdata, JJ, 
                         tempv1, tempv2, tempv3):
        cdef DTYPE_t jac_return = self.jac.evaluate(tt, yy, yp, cj, JJ)
        # TODO: convert jacmat to the jacobi matrix
        #raise NotImplemented
        return jac_return
          

    def __cinit__(self, residualfn, **options):
        """ Create the IDA Solver and initialize default values """
        
        self._ida_mem = IDACreate()
        
        default_values = {
            'implementation': 'serial',
            'use_relaxation': False,
            'rtol': 1e-6, 'atol': 1e-12,
            'linsolver': 'dense',
            'lband': 0,'uband': 0,
            'maxl': 0,
            'tcrit': 0.,
            'order': 5,
            'nsteps':500,
            'max_step': 0.,
            'first_step': 0.,
            'compute_initcond': None,
            'compute_initcond_t0': 0.01,
            'constraints_idx': None,
            'constraints_type': None,
            'algebraic_vars_idx':None, 
            'exclude_algvar_from_error': False,
            'user_data': None,
            'out': False, #
            'resfn': None
            }
 
        self.options = default_values
        self.N       = -1
        self.set_options(resfn=residualfn, **options)
        

    def set_options(self, **options):
        """ 
        Reads the options list and assigns values for the solver.
        
        Mandatory options are: 'resfn'.
        
        All options list:
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
                Values: 'dense' (= default), 'lapackdense', 'band', 'lapackband', 'spgmr', 'spbcg', 'sptfqmr'
                Description:
                    Specifies used linear solver.
                    Limitations: Linear solvers for dense and band matrices can be used only
                                 for serial implementation. For parallel implementation use_relaxation
                                 use lapackdense or lapackband respectively.
            'lband', 'uband':
                Values: non-negative integer, 0 = default
                Description:
                    Specifies the width of the lower band (below the main diagonal) and/or upper diagonal
                    (above the main diagonal). So the total band width is lband + 1 + uband, where the 1 
                    stands for the main diagonal (specially, lband = uband = 0 [that is the default] 
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
            'order': 
                Values: 1, 2, 3, 4, 5 (= default)
                Description:
                    Specifies the maximum order of the linear multistep method.
            'nsteps':
                Values: positive integer, 500 = default
                Description:
                    Maximum number of (internally defined) steps allowed during one call to the solver.
            'max_step': 
                Values: non-negative float, 0.0 = default
                Description:
                    Restricts the maximal (absolute) internal step value taken by the solver. The value of 0.0 uses
                    no restriction on the maximal step size.
            'first_step': 
                Values: float , 0.0 = default
                Description:
                    Sets the first step size. DAE solver can suffer on the first step, so set this
                    to circumvent this. The value of 0.0 uses the solver's internal default value.
            'compute_initcond':
                Values: 'y0', 'yp0', None
                Description:
                    Specifies what initial condition is calculated for given problem.
                    'y0'  - calculates/improves the y0 initial condition (considering yp0 to be accurate)
                    'yp0' - calculates/improves the yp0 initial condition (considering y0 to be accurate)
                    None  - don't compute any initial conditions (user provided values for y0 an yp0
                            are considered to be consistent and accurate)
            'compute_initcond_t0': 
                Values: non-negative float, 0.01 = default
                Description:
                    When calculating the initial condition, specifies the time until which the solver tries to
                    get the consistent values for either y0 or yp0.
            'user_data':
                Values: python object, None = default
                Description:
                    Additional data that are supplied to each call of the residual function 'resfn' (see below)
                    and Jacobi function 'jacfn' (if specified one).
            'resfn':
                Values: function of class ResFunction
                Description:
                    Defines the residual function (which has to be a subclass of ResFunction class). This function 
                    takes as input arguments current time t, current value of y, yp, numpy array of returned residual
                    and optional userdata. Return value is 0 if successfull.
                    This option is mandatory.
            'algebraic_vars_idx': 
                Values: numpy vector or None (= default)
                Description:
                    If given problem is of type DAE, some items of the residual vector returned by the 'resfn' have to
                    be treated as algebraic variables. These are denoted by the position (index) in the residual vector.
                    All these indexes have to be specified in the 'algebraic_vars_idx' array.
            'exclude_algvar_from_error': 
                Values: False (= default), True
                Description:
                    Indicates whether or not to suppress algebraic variables in the local error test. If 'algebraic_vars_idx'
                    vector not specified, this value is ignored.
                    
                    The use of this option (i.e. set to True) is discouraged when solving DAE systems of index 1, whereas it 
                    is generally encouraged for systems of index 2 or more.
            'constraints_idx':
                Values: numpy vector or None (= default)
                Description:
                    Constraint the individual variables. The variables are denoted by the position (index) in the residual vector.
                    All these indexes are/have to be specified in this 'constraints_idx' array.
            'constraints_type':
                Values: numpy vector or None (= default)
                Description:
                    The actuall type of contraints applied to the of the variables specified by 'constraints_idx'. The type is
                    one of:  0.0 - no contraint
                             1.0 - variable has to be non-negative (i.e. >= 0)
                             2.0 - variable has to be positive (i.e. > 0)
                            -1.0 - variable has to be non-positive (i.e. <= 0)
                            -2.0 - variable has to be negative (i.e. < 0)
            'out': False, #
        """

        for (key, value) in options.items():
            self.options[key.lower()] = value
            
    cpdef init_step(self, DTYPE_t t0, 
                    np.ndarray[DTYPE_t, ndim=1] y0, 
                    np.ndarray[DTYPE_t, ndim=1] yp0,
                    np.ndarray y_ic0_retn = None,
                    np.ndarray yp_ic0_retn = None):
        """ 
            Applies the set by 'set_options' method to the IDA solver.
            
            Performs computation of initial conditions (if compute_initcond flag set). Used only 
            with conjuction with the 'step' method to provide initialization and assure that the
            initial condition is calculated.
            
            Input:
                y0 - initial condition for y
                yp0 - initial condition for derivation of y
                y_ic0_retn - (optional) returns the calculated consistent initial condition for y
                yp_ic0_retn - (optional) returns the calculated consistent initial condition for 
                              y derivated
            
            Note: After setting (or changing) options with 'set_options' method you need to call 
                  'init_step' prior to the 'step' method to assure the values are correctly
                  initialized.
            
            Returns the time of the intial step.
        """
        #TODO: jacobi function ?isset
        #TODO: implement relaxation algoithm for opt: opts['use_relaxation']

        if self._ida_mem is NULL:
            raise MemoryError
 
        cdef dict opts = self.options
        self.parallel_implementation = (opts['implementation'].lower() == 'parallel')
     
        if opts['resfn'] == None:
            raise ValueError('The residual function ResFn not assigned '
                              'during ''set_options'' call !')
               
        if not len(y0) == len(yp0):
            raise ValueError('Arrays inconsistency: y0 and ydot0 have to be of the'
                             'same length !')
        if (((len(y0) == 0) and (not opts['compute_initcond'] == 'y0'))
                or ((len(yp0) == 0) and (not opts['compute_initcond'] == 'yp0'))):
            raise ValueError('Not passed y0 or ydot0 value has to computed'
                             'by ''init_cond'', but ''init_cond'' not set apropriately !')
        cdef long int N
        N = <long int>len(y0)
        
        cdef void* ida_mem = self._ida_mem

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
            self.y  = N_VClone(self.y0)
            self.yp = N_VClone(self.yp0)
 
        
        if self.N <= 0:
            IDAInit(ida_mem, _res, <realtype> t0, self.y0, self.yp0)
        elif self.N == N:
            IDAReInit(ida_mem, <realtype> t0, self.y0, self.yp0)
        else:
            # we should completely reallocate ida_mem, but...
            # probably also the res and jas functions should be changed
            # ==> add them to the (possible) options (set_options(...))
            raise NotImplemented('IDA object reallocation not implemented !')
        
        self.N = N
        
        # auxiliary variables
        self.aux_data = IDA_data(N)
        self.aux_data.parallel_implementation = self.parallel_implementation
        self.aux_data.res = opts['resfn']
        self.aux_data.user_data = opts['user_data']
        
        if not np.isscalar(opts['rtol']) :
            raise ValueError('rtol (%s) must be a scalar for IDA'\
                                % (opts['rtol']))
        cdef N_Vector atol
        cdef np.ndarray[DTYPE_t, ndim=1] np_atol
        cdef int maxl = <int> opts['maxl']
        if np.isscalar(opts['atol']):
            IDASStolerances(ida_mem, <realtype> opts['rtol'], <realtype> opts['atol'])
        else:
            np_atol = np.asarray(opts['atol'])
            if self.parallel_implementation:
                raise NotImplemented
            else:
                atol = N_VMake_Serial(N, <realtype *> np_atol.data)
            IDASVtolerances(ida_mem, <realtype> opts['rtol'], atol)
        #TODO: implement IDAFWtolerances(ida_mem, efun)
        
        # As user data we pass the (self) IDA object
        IDASetUserData(ida_mem, <void*> self.aux_data)
        
        cdef int order = <int> opts['order']
        if (order < 1) or (order > 5):
            raise ValueError('order should be betwen 1 and 5')
        IDASetMaxOrd(ida_mem, order)
        IDASetMaxNumSteps(ida_mem, <int> opts['nsteps'])
        if opts['max_step'] > 0.:
            IDASetMaxStep(ida_mem, <int> opts['max_step'])
        if opts['tcrit'] > 0.:
            IDASetStopTime(ida_mem, <realtype> opts['tcrit'])
        if opts['first_step'] > 0.:
            IDASetInitStep(ida_mem, <realtype> opts['first_step'])
        #TODO: implement the rest of IDASet* functions
        
        linsolver = opts['linsolver'].lower()
        if linsolver == 'dense':
            if self.parallel_implementation:
                raise ValueError('Linear solver for dense matrices can be used'
                                  'only for serial implementation. Use ''lapackdense''' 
                                  'instead for parallel implementation.')
            else:
                 IDADense(ida_mem, N)
        elif linsolver == 'lapackdense':
            IDALapackDense(ida_mem, N)
        elif linsolver == 'band':
            if self.parallel_implementation:
                raise ValueError('Linear solver for band matrices can be used'
                                  'only for serial implementation. Use ''lapackband'' '
                                  'instead for parallel implementation.')
            else:
                pass
                IDABand(ida_mem, N, <int> opts['uband'], <int> opts['lband']) 
        elif linsolver == 'lapackband':
            IDALapackBand(ida_mem, N, <int> opts['uband'], <int> opts['lband']) 
        elif linsolver == 'spgmr':
            IDASpgmr(ida_mem, maxl)
        elif linsolver == 'spbcg':
            IDASpbcg(ida_mem, maxl)
        elif linsolver == 'sptfqmr':
            IDASptfqmr(ida_mem, maxl)
            
        constraints_idx = opts['constraints_idx']
        constraints_type = opts['constraints_type']
        cdef unsigned int idx
        cdef np.ndarray[DTYPE_t, ndim=1] constraints_vars
        if not constraints_type is None:
            if not constraints_idx is None:
                constraints_vars = np.zeros(N, float)
                for idx in range(constraints_idx):
                    constraints_vars[constraints_idx[idx]] = constraints_type[idx]
            else:
                assert len(constraints_type) == N, 'Without ''constraints_idx'' specified the ''constraints_type'' has to be of the same length as y.'
                constraints_vars = np.asarray(constraints_type, float)
            if not self.constraints is NULL:
                N_VDestroy(self.constraints)
            if self.parallel_implementation:
                raise NotImplemented
            else:   
                self.constraints = N_VMake_Serial(N, <realtype*> constraints_vars.data)
            IDASetConstraints(ida_mem, self.constraints)
 
        cdef np.ndarray[DTYPE_t, ndim=1] dae_vars
        alg_vars_idx = opts['algebraic_vars_idx']
        compute_initcond = opts['compute_initcond']
        if (compute_initcond == 'yp0') or (opts['exclude_algvar_from_error'] and not alg_vars_idx is None):
            dae_vars = np.ones(N, float)
            if not alg_vars_idx is None:
                for algvar_idx in opts['algebraic_vars_idx']:
                    dae_vars[algvar_idx] = 0.0
            if not self.dae_vars_id is NULL:
                N_VDestroy(self.dae_vars_id)
            if self.parallel_implementation:
                raise NotImplemented
            else:
                self.dae_vars_id = N_VMake_Serial(N, <realtype*> dae_vars.data) 
            IDASetId(ida_mem, self.dae_vars_id)
            
            if opts['exclude_algvar_from_error'] and not alg_vars_idx is None:
                IDASetSuppressAlg(ida_mem, True)
        
        cdef int return_flag
        cdef float t0_init
        if compute_initcond == 'yp0':
            IDACalcIC(ida_mem, IDA_YA_YDP_INIT, <realtype>opts['compute_initcond_t0'])
            t0_init = opts['compute_initcond_t0']
        elif compute_initcond == 'y0':
            return_flag = IDACalcIC(ida_mem, IDA_Y_INIT, <realtype> opts['compute_initcond_t0'])
            if not (return_flag == IDA_SUCCESS):
                raise ValueError('IDA did not compute successfully the initial condition')
            t0_init = opts['compute_initcond_t0']
        elif compute_initcond == None or compute_initcond == '':
            t0_init = t0
        else: raise ValueError('Unknown ''compute_initcond'' calculation method: ''%s''' 
                                    %(compute_initcond))
        
        if not ((y_ic0_retn is None) and (yp_ic0_retn is None)):
            assert len(y_ic0_retn) == len(yp_ic0_retn) == N, 'y_ic0 and/or yp_ic0 have to be of the same size as y0.'
            return_flag = IDAGetConsistentIC(self._ida_mem, self.y, self.yp)
            if return_flag == IDA_ILL_INPUT:
                raise NameError('Method ''get_consistent_ic'' has to be called before'
                                'the first call to ''step'' method.')
            if not y_ic0_retn is None: nv_s2ndarray(self.y, y_ic0_retn)
            if not yp_ic0_retn is None: nv_s2ndarray(self.yp, yp_ic0_retn)
            
        #TODO: implement the rest of IDA*Set* functions for linear solvers

        #TODO: Rootfinding

        # TODO: useoutval, success    
        #self.useoutval = out
        #self.success = 1
        return t0_init
        
    def run_solver(self, np.ndarray[DTYPE_t, ndim=1] tspan, np.ndarray[DTYPE_t, ndim=1] y0, 
                   np.ndarray[DTYPE_t, ndim=1] yp0):
        """
        Runs the solver.
        
        Input:
            tspan - an numpy array of times at which the computed value will be returned
            y0    - numpy array of initial values
            yp0   - numpy array of initial values of derivatives
            
        Return values:
            flag   - indicating return status of the solver
            t      - numpy array of times at which the computations were successful
            y      - numpy array of values corresponding to times t (values of y[i, :] ~ t[i])
            yp     - numpy array of derivatives corresponding to times t (values of yp[i, :] ~ t[i])
            t_err  - float or None - if recoverable error occured (for example reached maximum
                     number of allowed iterations), this is the time at which it happened
            y_err  - numpy array of values corresponding to time t_err
            yp_err - numpy array of derivatives corresponding to time t_err
            
        Note:
            If 'calc_initcond' then solver returns instead of user supplied y0, yp0 values as
            the starting values the values calculated by the solver (i.e. consistent initial
            conditions. The starting time is then also the precomputed time.
        """
        cdef np.ndarray[DTYPE_t, ndim=2] y_retn, yp_retn
        y_retn  = np.empty([len(tspan), len(y0)], float)
        yp_retn = np.empty([len(tspan), len(y0)], float)
        cdef np.ndarray[DTYPE_t, ndim=1] t_retn
        t_retn  = np.empty([len(tspan), ], float)
        
        t_retn[0] = self.init_step(tspan[0], y0, yp0, y_retn[0, :], yp_retn[0, :])

        cdef DTYPE_t t
        cdef unsigned int idx
        #TODO: Parallel version
        cdef N_Vector y  = self.y
        cdef N_Vector yp = self.yp
        cdef realtype t_out
        cdef int flag
        cdef np.ndarray[DTYPE_t, ndim=1] y_err, yp_err

        for idx in range(len(tspan))[1:]:
            t = tspan[idx]
            flag = IDASolve(self._ida_mem, <realtype> t, &t_out, y, yp, IDA_NORMAL)
            self.solver_return_flag = flag
            if flag == -1 or flag == -2:
                print('An error occured. See ''solver_return_flag'' variable and documentation.')
                return np.asarray([]), np.asarray([])
                
            if flag < -3:
                t_retn  = np.asarray(t_retn[0:idx])
                y_retn  = np.asarray(y_retn[0:idx, :])
                yp_retn = np.asarray(yp_retn[0:idx, :])
                
                y_err = np.empty([self.N, ], float)
                yp_err = np.empty([self.N, ], float)
                nv_s2ndarray(y,  y_err) 
                nv_s2ndarray(yp, yp_err)
                
                t_retn[idx] = t_out
                return flag, t_retn, y_retn, yp_retn, t_out, y_err, yp_err
            else:
                t_retn[idx] = t_out
                nv_s2ndarray(y,  y_retn[idx, :]) 
                nv_s2ndarray(yp, yp_retn[idx, :]) 
            
                
        return flag, t_retn, y_retn, yp_retn, None, None, None

    def step(self, DTYPE_t t, np.ndarray[DTYPE_t, ndim=1] y_retn,
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
            yp_retn - numpy vector (ndim = 1) of None. If not None, will be
                      filled (needs to be preallocated)
                      with derivatives of y at time t.
        Return values:
            flag  - status of the computation (successful or error occured)
            t_out - time, where the solver stopped (when no error occured, t_out == t)
        """
        #TODO: check whether 'init_step' has been called
        cdef N_Vector y  = self.y
        cdef N_Vector yp = self.yp
        cdef realtype t_out
        cdef int flag
        if t>0.0:
            flag = IDASolve(self._ida_mem, <realtype> t, &t_out, y, yp, IDA_NORMAL)
        else:
            flag = IDASolve(self._ida_mem, <realtype> -t, &t_out, y, yp, IDA_ONE_STEP) 
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
        
