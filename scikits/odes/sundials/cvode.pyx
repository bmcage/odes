import numpy as np
cimport numpy as np

from c_sundials cimport realtype, N_Vector
from c_cvode cimport *
from common_defs cimport (nv_s2ndarray, ndarray2nv_s,
                          ndarray2DlsMatd,
                          RhsFunction, WrapRhsFunction)

cdef enum: HOOK_FN_STOP = 128

cdef int _rhsfn(realtype tt, N_Vector yy, N_Vector yp,
              void *auxiliary_data):
    """function with the signature of CVRhsFn"""
    cdef np.ndarray[DTYPE_t, ndim=1] yy_tmp, yp_tmp
    
    aux_data = <CV_data> auxiliary_data
    cdef bint parallel_implementation = aux_data.parallel_implementation

    if parallel_implementation:
        raise NotImplemented 
    else:
        yy_tmp = aux_data.yy_tmp
        yp_tmp = aux_data.yp_tmp
             
        nv_s2ndarray(yy, yy_tmp)
        nv_s2ndarray(yp, yp_tmp)
         
    aux_data.rhs.evaluate(tt, yy_tmp, yp_tmp, aux_data.user_data)

    return 0

cdef class CV_data:
    def __cinit__(self, N):
        self.parallel_implementation = False
        self.user_data = None
        
        self.yy_tmp = np.empty(N, float)
        self.yp_tmp = np.empty(N, float)
        self.jac_tmp = None

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
            # 'use_relaxation': False,
            'rtol': 1e-6, 'atol': 1e-12,
            'linsolver': 'dense',
            'lband': 0,'uband': 0,
            'maxl': 0,
            'precond_type': 'NONE',
            'tcrit': 0.,
            'order': 0,
            'max_step_size': 0.,
            'min_step_size': 0.,
            'first_step': 0.,
            'max_steps': 0,
            'bdf_stability_detection': False,
            'max_conv_fails': 0,
            'max_nonlin_iters': 0,
            'nonlin_conv_coef': 0.,
            'user_data': None,
            'rfn': None,
            # 'jacfn': None
            }
         
        self.options  = default_values
        self.N        = -1
        self.set_options(rfn=Rfn, **options)
        self._cv_mem = NULL
        self.initialized = False

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
                Values: function of class ResFunction or a python function with signature (t, y, yp, resultout)
                Description:
                    Defines the right-hand-side function (which has to be a subclass of ResFunction class, or a normal python function with signature (t, y, yp, resultout) ).
                    This function takes as input arguments current time t, current value of y, yp, numpy array of returned residual
                    and optional userdata. Return value is 0 if successfull.
                    This option is mandatory.
        """

        for (key, value) in options.items():
            self.options[key.lower()] = value
        self.initialized = False

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
            raise ValueError('CVODE:init: Unrecognized iter_type: %s' % iter_type)
        self.parallel_implementation = (opts['implementation'].lower() == 'parallel')
        cdef long int N
        N = <long int>len(y0)
        
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
            self.y0 = N_VMake_Serial(N, <realtype *>y0.data)
            self.y  = N_VClone(self.y0)

        cdef int flag
        cdef void* cv_mem = self._cv_mem
        if (cv_mem is NULL) or (self.N != N):
            if (not cv_mem is NULL):
                CVodeFree(&cv_mem)
            cv_mem = CVodeCreate(lmm, itert)
            if cv_mem is NULL:
                raise MemoryError('Could not create cv_mem object')
            
            self._cv_mem = cv_mem
            flag = CVodeInit(cv_mem, _rhsfn,  <realtype> t0, self.y0)
        elif self.N == N:
            flag = CVodeReInit(cv_mem, <realtype> t0, self.y0)
        else:
            raise ValueError('CVodeInit:Error: You should not be here...')
        if flag == CV_ILL_INPUT:
                raise ValueError('CVode[Re]Init: Ill input')
        elif (flag == CV_MEM_FAIL) or (flag == CV_MEM_NULL):
            raise MemoryError('CVode[Re]Init: Memory allocation error')
        elif flag == CV_NO_MALLOC:
            raise MemoryError('CVodeReInit: No memory allocated in CVInit.')

        self.N = N

        # auxiliary variables
        self.aux_data = CV_data(N)
        self.aux_data.parallel_implementation = self.parallel_implementation

        if not isinstance(opts['rfn'] , RhsFunction):
            tmpfun = WrapRhsFunction()
            tmpfun.set_rhsfn(opts['rfn'])
            opts['rfn'] = tmpfun
        self.aux_data.res = opts['rfn']
        #self.aux_data.jac = opts['jacfn']
        self.aux_data.user_data = opts['user_data']

        if not np.isscalar(opts['rtol']) :
            raise ValueError('rtol (%s) must be a scalar for CVode'
                                % (opts['rtol']))

        cdef N_Vector atol
        cdef np.ndarray[DTYPE_t, ndim=1] np_atol
        if not (self.atol is NULL):
            N_VDestroy(self.atol)
            self.atol = NULL
        if np.isscalar(opts['atol']):
            flag = CVodeSStolerances(cv_mem, <realtype> opts['rtol'], 
                                               <realtype> opts['atol'])
        else:
            np_atol = np.asarray(opts['atol'])
            if self.parallel_implementation:
                raise NotImplemented
            else:
                atol = N_VMake_Serial(N, <realtype *> np_atol.data)
                flag = CVodeSVtolerances(cv_mem, <realtype> opts['rtol'], atol)
        #TODO: implement CVFWtolerances(cv_mem, efun)
        
        if flag == CV_ILL_INPUT:
            raise ValueError('CVodeStolerances: negative atol or rtol value.')

        # As user data we pass the CV_data object
        CVodeSetUserData(cv_mem, <void*> self.aux_data)

        if (opts['order'] > 0):
            CVodeSetMaxOrd(cv_mem, <int> opts['order'])
        CVodeSetMaxNumSteps(cv_mem, <int> opts['max_steps'])
        if lmm_type == 'bdf':
            CVodeSetStabLimDet(cv_mem, <bint> opts['bdf_stability_detection'])
        CVodeSetInitStep(cv_mem, <realtype> opts['first_step'])
        if (opts['min_step_size'] > 0.):
           CVodeSetMinStep(cv_mem, <realtype> opts['min_step_size'])
        flag = CVodeSetMaxStep(cv_mem, <realtype> opts['max_step_size'])
        if flag == CV_ILL_INPUT:
            raise ValueError('CVodeSetMaxStep: max_step_size is negative or smaller than min_step_size.')
        if opts['tcrit'] > 0.:
            CVodeSetStopTime(cv_mem, <realtype> opts['tcrit'])
        if opts['max_nonlin_iters'] > 0:
            CVodeSetMaxNonlinIters(cv_mem, <int> opts['max_nonlin_iters'])
        if opts['max_conv_fails'] > 0:
            CVodeSetMaxConvFails(cv_mem, <int> opts['max_conv_fails'])
        if opts['nonlin_conv_coef'] > 0:
            CVodeSetNonlinConvCoef(cv_mem, <int> opts['nonlin_conv_coef'])

        if iter_type == 'newton':
            if self.parallel_implementation:
                raise NotImplementedError
                
            linsolver = opts['linsolver'].lower()

            if linsolver == 'dense':
                if self.parallel_implementation:
                    raise ValueError('Linear solver for dense matrices can be' 
                                     'used only for serial implementation. Use'
                                     ' ''lapackdense'' for parallel implementation.')
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
                    raise ValueError('CVLapackDense solver is not compatible with'
                                         ' the current nvector implementation.')
                elif flag == CVDLS_MEM_FAIL:
                    raise MemoryError('CVLapackDense memory allocation error.')
            elif linsolver == 'band':
                flag = CVBand(cv_mem, N, <int> opts['uband'], 
                                         <int> opts['lband'])
                if flag == CVDLS_ILL_INPUT:
                     raise ValueError('CVBand solver is not compatible'
                                      ' with the current nvector implementation'
                                      ' or bandwith outside range.')
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
                        raise MemoryError('LinSolver:CVSpils memory allocation error.')
            else:
                raise ValueError('LinSolver: Unknown solver type: %s'
                                     % opts['linsolver'])

        #TODO: Rootfinding

        self.initialized = True

        return t0

    def solve(self, object tspan, object y0, object hook_fn = None):

        cdef np.ndarray[DTYPE_t, ndim=1] np_tspan, np_y0
        np_tspan = np.asarray(tspan)
        np_y0    = np.asarray(y0)
        #TODO: determine hook_fn type

        return self._solve(np_tspan, np_y0, hook_fn)
            
    cpdef _solve(self, np.ndarray[DTYPE_t, ndim=1] tspan, 
                 np.ndarray[DTYPE_t, ndim=1] y0, hook_fn = None):

        cdef np.ndarray[DTYPE_t, ndim=1] t_retn
        cdef np.ndarray[DTYPE_t, ndim=2] y_retn
        t_retn  = np.empty([np.shape(tspan), ], float)
        y_retn  = np.empty([np.alen(tspan), np.alen(y0)], float)
            
        self._init_step(tspan[0], y0)

        cdef np.ndarray[DTYPE_t, ndim=1] y_last
        cdef unsigned int idx
        cdef DTYPE_t t, t_onestep
        cdef int flag
        cdef void *cv_mem = self._cv_mem
        cdef realtype t_out
        cdef N_Vector y  = self.y
        cdef bint _flag

        y_last   = np.empty(np.shape(y0), float)
                
        if hook_fn:
            for idx in np.arange(len(tspan))[1:]:
                t = tspan[idx]

                while True:
                    flag = CVode(cv_mem, <realtype> t,  y, &t_out, 
                                   CV_ONE_STEP)

                    nv_s2ndarray(y,  y_last)
                    
                    if ((flag < 0) or 
                        hook_fn(t_out, y_last, self.aux_data.user_data) != 0):
                        if flag < 0:
                            print('Error occured. See ''solver_return_flag'' '
                                  'variable and CVode documentation.')
                        else:
                            flag = HOOK_FN_STOP

                        t_retn  = t_retn[0:idx]
                        y_retn  = y_retn[0:idx, :]
                
                        return flag, t_retn, y_retn, t_out, y_last

                    if t_out > t: break

                t_retn[idx]    = t_out 
                y_retn[idx, :] = y_last
        else:
            for idx in np.arange(len(tspan))[1:]:
                t = tspan[idx]

                flag = CVode(cv_mem, <realtype> t,  y, &t_out, CV_NORMAL)

                nv_s2ndarray(y,  y_last)

                if flag < 0:
                    print('Error occured. See ''solver_return_flag'' '
                          'variable and CVode documentation.')
                    t_retn  = t_retn[0:idx]
                    y_retn  = y_retn[0:idx, :]
                
                    return flag, t_retn, y_retn, t_out, y_last

                t_retn[idx]    = t_out
                y_retn[idx, :] = y_last
                    
        return flag, t_retn, y_retn, None, None

    def step(self, DTYPE_t t, np.ndarray[DTYPE_t, ndim=1] y_retn):
        if not self.initialized:
            raise ValueError('Method ''init_step'' has to be called prior to the'
                             'first call of ''step'' method.')

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

           
            
            
