# Authors: B. Malengier
"""
First-order DAE solver
"""

from __future__ import print_function

integrator_info = \
"""
ddaspk
~~~~~~
Solver developed 1989 to 1996, with some corrections from 2000 - Fortran

This code solves a system of differential/algebraic equations of the form 
G(t,y,y') = 0 , using a combination of Backward Differentiation Formula 
(BDF) methods and a choice of two linear system solution methods: direct 
(dense or band) or Krylov (iterative). 
Krylov is not supported from within scikits.odes. 
In order to support it, a new interface should be created ddaspk_krylov, 
with a different signature, reflecting the changes needed.

Source: http://www.netlib.org/ode/ddaspk.f

On construction the function calculating the residual (res) must be given and 
optionally also the function calculating the jacobian (jac). 
res has the signature: res(x, y, yprime, return_residual)
with 
    x       : independent variable, eg the time, float
    y       : array of n unknowns in x
    yprime  : dy/dx array of n unknowns in x, dimension = dim(y)
    return_residual: array that must be updated with the value of the residuals,
              so G(t,y,y').  The dimension is equal to dim(y)
    return value: Not needed. However, for use with other solvers, consider 
              returning 0 on success.

jac has the signature jac(x, y, yprime, cj, return_jac)
with 
    x       : independent variable, eg the time, float
    y       : array of n unknowns in x
    yprime  : dy/dx array of n unknowns in x, dimension = dim(y)
    cj      : internal variable of ddaspk algorithm you can use, don't change it! 
              cj can be ignored, or used to rescale constraint equations in the 
              system
    return_jac : two dimensional array of the Jacobian, as per the Jacobian 
              definition of the ddaspk solver. This means it should be or a 
                full nxn shaped array in general (n=dim(y)), or a banded shaped
                array as per the definition of lband/uband. 
                Jac is optional and should be set with the jacfn option keyword. 
                Note that jac is defined as 
                            dres(i)/dy(j) + cj*dres(i)/dyprime(j)
    return value: Not needed. However, for use with other solvers, consider 
              returning 0 on success.

This integrator accepts the following parameters in the initializer or 
set_options method of the dae class:

- rfn : residual function, see above for signature. This option need not be
        set, as rfn will be set during initialization. If the residual function
        of initialization needs to be reset, this option can be used
- jacfn : jacobian function, see above for signature. Default is None.
- atol : float or sequence of length i
  absolute tolerance for solution
- rtol : float or sequence of length i
  relative tolerance for solution
- lband : None or int
- uband : None or int
  Jacobian band width, jac[i,j] != 0 for i-lband <= j <= i+uband.
  Setting these requires your jac routine to return the jacobian
  in packed format, jac_packed[i-j+lband, j] = jac[i,j].
- tcrit : None or float. If given, tcrit is a critical time point
  beyond which no integration occurs
- max_steps : int
  Maximum number of (internally defined) steps allowed during one
  call to the solver.
- first_step : float
  Set initial stepsize. DAE solver can suffer on the first step, set this
  to circumvent this.
- max_step_size : float
  Limits for the step sizes used by the integrator. This overrides the
  internal value
- order : int
  Maximum order used by the integrator, >=1,  <= 5 for BDF.
  5 is the default
- enforce_nonnegativity: bool
  Enforce the nonnegativity of Y during integration
  Note: best is to run code first with no condition
- compute_initcond: None or 'yp0' or 'y0'
  DDASPK may be able to compute the initial conditions if you do not know them
  precisely. 
  If y0, then y0 will be calculated
  If yp0, then the differential variables will be used to solve for the 
    algebraic variables and the derivative of the differential variables. 
    Which are the algebraic variables must be indicated with algebraic_var method
- exclude_algvar_from_error: bool
  To determine solution, do not take the algebraic variables error control into 
  account. Default=False
- constraint_init: bool
  Enforce the constraint checks of Y during initial condition computation
  Note: try first with no constraints
- constraint_type: if constraint_init, give an integer array with for every
  unknown the specific condition to check: 
       1: y0[i] >= 0 
       2: y0[i] >  0
      -1: y0[i] <= 0
      -2: y0[i] <  0
       0: y0[i] not constrained
  Alternatively, pass only one integer that then applies to all unknowns
- algebraic_vars_idx: an array or None (= default)
                Description:
                    If given problem is of type DAE, some items of the residual
                    vector returned by the 'resfn' have to be treated as 
                    algebraic variables. These are denoted by the position 
                    (index) in the residual vector.
                    All these indexes have to be specified in the 
                    'algebraic_vars_idx' array.
"""

__doc__ += integrator_info

__all__ = ['ddaspk']
__version__ = "$Id$"
__docformat__ = "restructuredtext en"

from numpy import asarray, array, zeros, sin, int32, isscalar, empty, alen
from copy import copy
from .dae import DaeBase
import re, sys

class ddaspk(DaeBase):
    __doc__ += integrator_info
    
    try:
        from .ddaspk import ddaspk as _runner
    except ImportError:
        print(sys.exc_info()[1])
        _runner = None
    # _runner = getattr(_ddaspk,'ddaspk',None)
    name = 'ddaspk'

    messages = { 1: 'A step was successfully taken in the '
                    'intermediate-output mode.  The code has not '
                    'yet reached TOUT.', 
                 2: 'The integration to TSTOP was successfully '
                    'completed (T = TSTOP) by stepping exactly to TSTOP.', 
                 3: 'The integration to TOUT was successfully '
                    'completed (T = TOUT) by stepping past TOUT. '
                    'Y(*) and YPRIME(*) are obtained by interpolation.', 
                 4: 'The initial condition calculation, with '
                    'INFO(11) > 0, was successful, and INFO(14) = 1. '
                    'No integration steps were taken, and the solution '
                    'is not considered to have been started.', 
                -1: 'A large amount of work has been expended (about 500 steps)',
                -2: 'Excess accuracy requested. (Tolerances too small.)',
                -3: 'The local error test cannot be satisfied because you '
                    'specified a zero component in ATOL and the corresponding'
                    ' computed solution component is zero.  Thus, a pure'
                    ' relative error test is impossible for this component.',
                -5: 'Repeated failures in the evaluation or processing of the'
                    ' preconditioner (in JAC)',
                -6: 'repeated error test failures on the last attempted step)', 
                -7: 'The nonlinear system solver in the time integration could'
                    ' not converge.',
                -8: 'The matrix of partial derivatives appears to be singular'
                    ' (direct method).', 
                -9: 'The nonlinear system solver in the time integration'
                    'failed to achieve convergence, and there were repeated '
                    'error test failures in this step.', 
                -10:'The nonlinear system solver in the time integration failed'
                    ' to achieve convergence because IRES was equal to -1.', 
                -11:'IRES = -2 was encountered and control is'
                    'being returned to the calling program.', 
                -12:'Failed to compute the initial Y, YPRIME.', 
                -13:"Unrecoverable error encountered inside user's"
                    "PSOL routine, and control is being returned to"
                    "the calling program.", 
                -14:'The Krylov linear system solver could not '
                    'achieve convergence.', 
                -33:'The code has encountered trouble from which'
                   ' it cannot recover.  A message is printed'
                   ' explaining the trouble and control is returned'
                   ' to the calling program.  For example, this occurs'
                   ' when invalid input is detected.', 
                }
    supports_run_relax = 0
    supports_step = 1

    def __init__(self, resfn, **options):
        default_values = {
            'rtol': 1e-6,
            'atol': 1e-12,
            'lband': None,
            'uband': None,
            'tcrit': None, 
            'order' : 5,
            'max_steps' : 500,
            'max_step_size' : 0.0, # corresponds to infinite
            'first_step' : 0.0, # determined by solver
            'enforce_nonnegativity': False, 
            'nonneg_type': None, 
            'compute_initcond': None,
            'constraint_init': False, 
            'constraint_type': None, 
            'algebraic_vars_idx': None,
            'exclude_algvar_from_error': False, 
            'rfn': None,
            'jacfn': None,
            }
        self.t = None
        self.y = None
        self.yp = None
        self.tmp_res = None
        self.tmp_jac = None
        self.options = default_values
        self.set_options(rfn=resfn, **options)
        self.initialized = False

    def set_options(self, **options):
        """ See dae.DaeBase
        """
        for (key, value) in options.items():
            self.options[key.lower()] = value
        self.initialized = False
        
    def _init_data(self):
        self.rtol = self.options['rtol']
        self.atol = self.options['atol']
        self.mu = self.options['uband']
        self.ml = self.options['lband']
        self.jac = self.options['jacfn']
        self.res = self.options['rfn']

        self.tcrit = self.options['tcrit']
        if self.options['order'] > 5 or self.options['order'] < 1:
            raise ValueError('order should be >=1, <=5')
        self.order = self.options['order']
        self.nsteps = self.options['max_steps']
        self.max_step = self.options['max_step_size']
        self.first_step = self.options['first_step']
        self.nonneg =0 
        if self.options['enforce_nonnegativity'] and self.options['constraint_init']: 
            self.nonneg = 3
        elif self.options['enforce_nonnegativity']: 
            self.nonneg = 2
        elif self.options['constraint_init']: 
            self.nonneg = 1
        if (self.nonneg == 1 or self.nonneg == 3) and self.options['constraint_type'] is None:
            raise ValueError('Give type of init cond contraint as '\
                              'an int array (>=0, >0, <=0, <0) or as int')
        else: self.constraint_type = self.options['constraint_type']
        if self.options['compute_initcond'] is None: 
            self.compute_initcond = 0
        elif re.match(self.options['compute_initcond'], r'y0', re.I): 
            self.compute_initcond = 2
        elif re.match(self.options['compute_initcond'], r'yp0', re.I): 
            self.compute_initcond = 1
        else: 
            raise ValueError('Unknown init cond calculation method %s' %(
                                            self.options['compute_initcond']))
        if self.compute_initcond == 1 and not self.options['algebraic_vars_idx']:
            raise ValueError('Give array indicating which are the '\
                    'algebraic variables with the algebraic_vars_idx option')
        self.algvaridx = self.options['algebraic_vars_idx']
        self.excl_algvar_err = self.options['exclude_algvar_from_error']

        self.success = 1
    
    def init_step(self, t0, y0, yp0, y_ic0_retn = None, yp_ic0_retn = None):
        """ See dae.DaeBase
        """
        self._init_data()
        self.y0 = y0
        self.yp0 = yp0
        self.t = t0
        self._reset(len(y0), self.options['jacfn'] is not None)
        
        if self.compute_initcond:
            if self.first_step <= 0.:
                first_step = 1e-18
            else:
                first_step = self.first_step
            self.y, self.yp, t = self.__run([3, 4], y0, yp0, t0, t0 + first_step)
            t0_init = t
        else:
            self.y = copy(y0)
            self.yp = copy(yp0)
            t0_init = t0
        if not y_ic0_retn is None: y_ic0_retn[:] = self.y[:]
        if not yp_ic0_retn is None: yp_ic0_retn[:] = self.yp[:]

        self.initialized = True
        return t0_init

    def _reset(self, n, has_jac):
        # Calculate parameters for Fortran subroutine ddaspk.
        self.neq = n
        self.algebraic_var = [1]*self.neq
        if self.algvaridx is not None:
            for ind in self.algvaridx:
                self.algebraic_var[ind] = -1
        self.info = zeros((20,), int32)  # default is all info=0
        self.info[17] = 2  # extra output on init cond computation
        if (isscalar(self.atol) != isscalar(self.rtol)) or (
               not isscalar(self.atol) and len(self.atol) != len(self.rtol)):
            raise ValueError('atol (%s) and rtol (%s) must be both scalar or'\
                    ' both arrays of length %s' % (self.atol, self.rtol, n))
        if not isscalar(self.atol):
            self.info[1] = 1
        if has_jac:
            self.info[4] = 1
        if self.mu is not None or self.ml is not None:
            if self.mu is None: self.mu = 0
            if self.ml is None: self.ml = 0
            self.info[5] = 1
        if self.excl_algvar_err:
            self.info[15] = 1
        lrw = 50 + max(self.order+4,7)*n
        if self.info[5]==0: lrw += pow(n, 2)
        elif self.info[4]==0: 
            lrw += (2*self.ml+self.mu+1)*n + 2*(n/(self.ml+self.mu+1)+1)
        else: lrw += (2*self.ml+self.mu+1)*n
        if self.info[15] == 1: lrw +=n
        rwork = zeros((lrw,), float)
        liw = 40 + n
        if self.nonneg in [1, 3]: liw += n
        if self.compute_initcond or self.excl_algvar_err: liw += n
        iwork = zeros((liw,), int32)
        if self.tcrit is not None:
            self.info[3] = 1
            rwork[0] = self.tcrit
        if self.max_step > 0.0 :
            self.info[6] = 1
            rwork[1] = self.max_step
        if self.first_step > 0.0 :
            self.info[7] = 1
            rwork[2] = self.first_step
        self.rwork = rwork
        if self.ml is not None:
            iwork[0] = self.ml
        if self.mu is not None:
            iwork[1] = self.mu
        if self.order < 5 :
            self.info[8] = 1
            iwork[2] = self.order
        iwork[5] = self.nsteps
        iwork[6] = 2           # mxhnil
        self.info[9] = self.nonneg
        lid = 40
        if self.info[9]==1 or self.info[9]==3 :
            lid = 40 + n
            if isscalar(self.constraint_type): 
                iwork[40:lid]=self.constraint_type
            else: iwork[40:lid]=self.constraint_type[:]
        self.info[10]=self.compute_initcond
        if self.info[10] in [1, 2]:
            iwork[lid:lid+n] = self.algebraic_var[:]
        if self.excl_algvar_err:
            iwork[lid:lid+n] = self.algebraic_var[:]
        ## some overrides that one might want
        # self.info[17] = 1  # minimal printing inside init cond calc
        # self.info[17] = 2  # full printing inside init cond calc
        if self.tcrit is not None:
            self.info[3] = 1
            self.rwork[0] = self.tcrit
        else:
            self.info[3] = 0
            self.rwork[0] = 0.

        self.iwork = iwork
        
        self.call_args = [self.info,self.rtol,self.atol,self.rwork,self.iwork]        #create storage
        self.tmp_res = empty(self.neq, float)
        if (self.ml is None or self.mu is None) :
            self.tmp_jac = zeros((self.neq, self.neq), float)
        else:
            self.tmp_jac = zeros((self.ml + self.mu + 1, self.neq), float)

        self.success = 1

    def _resFn(self, t, y, yp):
        """Wrapper around the residual as defined by user, to create
         the residual needed by ddaspk
        """
        self.res(t, y, yp, self.tmp_res)
        return self.tmp_res

    def _jacFn(self, t, y, yp, cj):
        """Wrapper around the jacobian as defined by user, to create
         the jacobian needed by ddaspk
        """
        self.jac(t, y, yp, cj, self.tmp_jac)
        return self.tmp_jac

    def solve(self, tspan, y0,  yp0, hook_fn = None):
        """ See dae.DaeBase
        """
        
        t_retn  = empty([alen(tspan), ], float)
        y_retn  = empty([alen(tspan), alen(y0)], float)
        yp_retn = empty([alen(tspan), alen(y0)], float)
        
        y_ic0_retn  = empty(alen(y0), float)
        yp_ic0_retn  = empty(alen(y0), float)
        tinit = self.init_step(tspan[0], y0, yp0, y_ic0_retn, yp_ic0_retn)
        
        t_retn[0] = tinit
        y_retn[0,:] = y0[:]
        yp_retn[0, :] = yp0[:]
        indbreak = None
        for ind, time in enumerate(tspan[1:]):
            if not self.success:
                indbreak = ind + 1
                break
            result = self.__run([2, 3], y_retn[ind], yp_retn[ind], t_retn[ind], time)
            t_retn[ind+1] = result[2]
            y_retn[ind+1][:] = result[0][:]
            yp_retn[ind+1][:] = result[1][:]
        self.t = t_retn[-1]
        if indbreak is not None:
            self.t = t_retn[indbreak-1]
            return self.success, t_retn[:indbreak], y_retn[:indbreak],\
                   yp_retn[:indbreak], t_retn[indbreak], y_retn[indbreak],\
                   yp_retn[indbreak]
        return self.success, t_retn, y_retn, yp_retn, None, None, None

    def __run(self, states, *args):
        # args are: y0,yprime0,t0,t1,res_params,jac_params
        y1, y1prime, t, self.flag = self._runner(*( (self._resFn, self._jacFn) \
                                           + args[:4] + tuple(self.call_args)))
        if self.flag < 0:
            print('ddaspk:',self.messages.get(self.flag,
                                        'Unexpected istate=%s' % self.flag))
            self.success = 0
        elif self.flag not in states:
            print('ddaspk: Run successfull. Unexpected istate=%s, stopping' %
                                            self.flag)
            print(self.messages.get(self.flag, 'Unknown istate=%s' % self.flag))
            self.success = 0
        return y1, y1prime, t

    def step(self, t, y_retn, yp_retn = None):
        """ See dae.DaeBase
        """
        if not self.initialized:
            raise ValueError('Method ''init_step'' has to be called prior to the'
                    'first call of ''step'' method, or after changing options')

        if t > 0.0:
            self.y, self.yp, self.t = self.__run([2, 3], self.y, self.yp, self.t, t)
        else:
            self.info[2] = 1
            self.y, self.yp, self.t = self.__run([1, 2], self.y, self.yp, self.t, -t)
            self.info[2] = 0
        y_retn[:] = self.y[:]
        if yp_retn is not None:
            yp_retn[:] = self.yp[:]
        
        return self.flag, self.t

if ddaspk._runner:
    DaeBase.integrator_classes.append(ddaspk)
