import numpy as np
cimport numpy as np

from c_sundials cimport realtype, N_Vector
from c_ida cimport *
from common_defs cimport *

#try:
#    cimport c_ida
#    cimport c_nvector_serial as c_nvecserial
#except:
#     print("Warning: ida solver not available, sundials wrapper package needed")
#     raise ImportError

#from c_ida import N_Vector     

# TODO: parallel implementation: N_VectorParallel
# TODO: linsolvers: check the output value for errors
# TODO: flag for indicating the resfn (in set_options) whether is a c-function or python function
# TODO: implement event handler

cdef int _res(realtype tt, N_Vector yy, N_Vector yp,
              N_Vector rr, void *auxiliary_data):

    #cdef np.ndarray[DTYPE_t, ndim=1] residual_tmp, yy_tmp, yp_tmp
    
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
         
    #TODO: probably a bottleneck, because self.res is not typed
    #      so the compiler will not optimize it... but otoh, in 
    #      general it is a python function...:)
    #TODO: pass user data to the function
    #      (add to 'set_options' option 'user_data' and pass it
    #      to the user "res" function
    aux_data.res.evaluate(tt, yy_tmp, yp_tmp, residual_tmp)
         
    if parallel_implementation:
        raise NotImplemented 
    else:
        ndarray2nv_s(rr, residual_tmp)
         
    return 0

cdef class IDA:
    cdef realtype _jacFn(self, Neq, tt, yy, yp, resvec, cj, jdata, JJ, 
                         tempv1, tempv2, tempv3):
        cdef DTYPE_t jac_return = self.jac.evaluate(tt, yy, yp, cj, JJ)
        # TODO: convert jacmat to the jacobi matrix
        #raise NotImplemented
        return jac_return
          

    def __cinit__(self):
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
            'compute_initcond': '',
            'compute_initcond_t0': 0.01,
            'constraints': False,        #
            'constraint_type': None,     # 
            'algebraic_vars_idx': np.array([]), 
            'exclude_algvar_from_error': False,   #
            'out': False, #
            'resfn': None
            }
        
        #NOTE: 1. max_step=0.0 corresponds to infinite
        #      2. first_step =. 0.0 => first step will be determined by solver
        #      3. possible linsolvers: 'dense', 'lapackdense', 'band', 'labackband',
        #                              'spgmr', 'spbcg', 'sptfqmr'
        #      4. maxl - dimension of the number of used Krylov subspaces (0 = default)
        #              (used by 'spgmr', 'spbcg', 'sptfqmr' linsolvers)
        #      5. compute_initcond: '', 'yode0', 'yprime0'
        
        
        self.options = default_values
        self.N       = -1
        
        #np.ndarray rtol=1e-6, np.ndarray atol=1e-12,
        #lband=None,uband=None,
        #tcrit=None, 
        #int order = 5,
        #int nsteps = 500,
        #double max_step = 0.0, # corresponds to infinite
        #double first_step = 0.0, # determined by solver
        #char* compute_initcond='',
        #double compute_initcond_t0 = 0.01,
        #constraints=False, 
        #constraint_type=None, 
        #algebraic_vars_idx=None, 
        #exclude_algvar_from_error=False,
        #out = False
        
        
          
    def set_options(self, **options):
        """ 
        Reads the options list and assigns values for the solver.
        
        Mandatory options are: 'resfn'.
        
        All options (and their defaults) are:
            'implementation': 'serial'
            'use_relaxation': False
            'rtol': 1e-6, 'atol': 1e-12
            'linsolver': 'dense'
            'lband': 0,'uband': 0
            'maxl': 0
            'tcrit': 0.
            'order': 5,
            'nsteps':500
            'max_step': 0.
            'first_step': 0.
            'compute_initcond': '',
            'compute_initcond_t0': 0.01
            'constraints': False,        #
            'constraint_type': None,     # 
            'algebraic_vars_idx': np.array([]), 
            'exclude_algvar_from_error': False,   #
            'out': False, #
            'resfn': None
        """

        for (key, value) in options.items():
            self.options[key.lower()] = value
            
    cpdef init_step(self, DTYPE_t t0, 
                 np.ndarray[DTYPE_t, ndim=1] y0, 
                 np.ndarray[DTYPE_t, ndim=1] ydot0):
        """ 
            Applies the set by 'set_options' method to the IDA solver.
            
            Performs computation of initial conditions (if compute_initcond flag set). Used only 
            with conjuction with the 'step' method to provide initialization and assure that the
            initial condition is calculated.
            
            Note: After setting (or changing) options with 'set_options' method you need to call 
                  'init_step' prior to the 'step' method to assure the values are correctly
                  initialized.
            
            Returns the time of the intial step.
        """
        #TODO: jacobi function ?isset
        #TODO: implement relaxation algoithm for opt: opts['use_relaxation']
        #TODO: return also the values of y at the initial time

        if self._ida_mem is NULL:
            raise MemoryError
 
        cdef dict opts = self.options
        print('Options: ')
        print(opts)
        print()
        self.parallel_implementation = (opts['implementation'].lower() == 'parallel')
     
        if opts['resfn'] == None:
            raise ValueError('The residual function ResFn not assigned '
                              'during ''set_options'' call !')
               
        if not len(y0) == len(ydot0):
            raise ValueError('Arrays inconsistency: y0 and ydot0 have to be of the'
                             'same length !')
        if (((len(y0) == 0) and (not opts['compute_initcond'] == 'yode0'))
                or ((len(ydot0) == 0) and (not opts['compute_initcond'] == 'yprime0'))):
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
            self.yp0 = N_VMake_Serial(N, <realtype *>ydot0.data)
            self.residual = N_VNew_Serial(N)
            self.y  = N_VClone(self.y0)
            self.yp = N_VClone(self.yp0)
            self.aux_data = IDA_data()
        
        # auxiliary variables
        self.aux_data.yy_tmp = np.empty(N, float)
        self.aux_data.yp_tmp = np.empty(N, float)
        self.aux_data.residual_tmp = np.empty(N, float)
        self.aux_data.parallel_implementation = self.parallel_implementation
        self.aux_data.res = opts['resfn']
        
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
        
        if not np.isscalar(opts['rtol']) :
            raise ValueError('rtol (%s) must be a scalar for IDA'\
                                % (opts['rtol']))
        #cdef np.ndarray atol
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
#        elif linsolver == 'spgmr':
#            IDASpgmr(ida_mem, maxl)
        elif linsolver == 'spbcg':
            #cdef int maxl = <int> opts['maxl']
            IDASpbcg(ida_mem, maxl)
        elif linsolver == 'sptfqmr':
            #cdef int maxl = <int> opts['maxl']
            IDASptfqmr(ida_mem, maxl)
        
        cdef np.ndarray[DTYPE_t, ndim=1] dae_vars = np.ones(N, float)
        cdef int return_flag
        cdef float t0_init
        compute_initcond = opts['compute_initcond']
        if compute_initcond == 'yode0':
            for algvar_idx in opts['algebraic_vars_idx']:
                dae_vars[algvar_idx] = 0.0
            if not self.dae_vars_id is NULL:
                N_VDestroy(self.dae_vars_id)
            if self.parallel_implementation:
                raise NotImplemented
            else:
                self.dae_vars_id = N_VMake_Serial(N, <realtype*> dae_vars.data) 
            IDASetId(ida_mem, self.dae_vars_id)
            IDACalcIC(ida_mem, IDA_YA_YDP_INIT, <realtype>opts['compute_initcond_t0'])
        elif compute_initcond == 'yprime0':
            return_flag = IDACalcIC(ida_mem, IDA_Y_INIT, <realtype> opts['compute_initcond_t0'])
            if not (return_flag == IDA_SUCCESS):
                raise ValueError('IDA did not compute successfully the initial condition')
            t0_init = opts['compute_initcond_t0']
        elif compute_initcond == '':
            t0_init = t0
        else: raise ValueError('Unknown ''compute_initcond'' calculation method: ''%s''' 
                                    %(compute_initcond))
        #TODO: implement the rest of IDA*Set* functions for linear solvers

        #TODO: Rootfinding

        #TODO: Constraints
#            elif key == 'constraints':
                # TODO: constraints, constraint_type 
                #if constraints and constraint_type is None:
                #    raise ValueError('Give type of contraint as '\
                #                      'an array (1:>=0, 2:>0, -1:<=0, -2:<0)')
                #elif constraints:
                #    self.constraint_type = nvecserial.NVector(list(constraint_type))
                #else:
                #    self.constraint_type = None

        #self.excl_algvar_err = exclude_algvar_from_error
        
        
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
            numpy array of computed values at time specified by 'tspan' input argument
        """
        #TODO: what if an error occures? what is the returned value then?
        #      should we return also the time at which error occured?
        self.init_step(tspan[0], y0, yp0)
        
        cdef np.ndarray[DTYPE_t, ndim=2] y_retn
        #TODO: store also yp - add another option "store_yp" to return it if needed
        y_retn = np.empty([len(tspan), len(y0)], float)
        cdef DTYPE_t t
        cdef unsigned int idx
        #TODO: Parallel version
        cdef N_Vector y  = self.y
        cdef N_Vector yp = self.yp
        cdef realtype t_out
        y_retn[0,:] = y0
        for idx in range(len(tspan))[1:]:
            t = tspan[idx]
            IDASolve(self._ida_mem, <realtype> t, &t_out, y, yp, IDA_NORMAL)
            nv_s2ndarray(y, y_retn[idx, :])
            
        return y_retn

    def step(self, realtype t):
        """
        Method for calling successive next step of the IDA solver to allow
        more precise control over the IDA solver. The 'init_step' method has to
        be called before the 'step' method.
        
        Input:
            t - time (scalar) when next values are computed
        Return values:
            array of computed values at time 't'
        """
        #TODO: implement next step
        #TODO: check whether 'init_step' has been called
        raise NotImplemented
        pass

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
        
