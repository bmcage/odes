# Authors: B. Malengier based on ode.py
"""
First-order DAE solver

User-friendly interface to various numerical integrators for solving an
algebraic system of first order ODEs with prescribed initial conditions:

         d y(t)
    A * ---------  = f(t,y(t)),
            d t

    y(t=0)[i] = y0[i],
    
      d y(t=0)
    ---------- [i]  = yprime0[i],
        d t

where::

    i = 0, ..., len(y0) - 1
    A is a (possibly singular) matrix of size i x i
    f(t,y) is a vector of size i
    
or more generally, equations of the form 

    G(t,y,y') = 0

class dae
---------

A generic interface class to differential algebraic equations. 
It has the following methods::

    integrator = dae(integrator_name, resfn, **options)
    integrator.set_options(options)
    result = integrator.solve(times, init_val_y, init_val_yp, user_data)

Alternatively, an init_step, and step method can be used to iterate over a 
solution.

For dae resfn is required, this is the residual equations evaluator
function, which must satisfy a specific signature.
"""

from __future__ import print_function

integrator_info_ddaspk = \
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
res has the signature: res(x, y, yprime, cj)
with 
    x : independent variable, eg the time, float
    y : array of n unknowns in x
    yprime : dy/dx array of n unknowns in x
    cj : internal variable of ddaspk algorithm you can use, don't change it! 
         cj can be ignored, or used to rescale constraint equations in the 
         system
return value should be an array with the result of the residual computation

jac has the signature jac(x, y, yprime, cj) as res, however the return value 
should be a nxn shaped array in general or a banded shaped array as per the
definition of lband/uband below. Jac is optional. 
Note that jac is defined as dres(i)/dy(j) + cj*dres(i)/dyprime(j)

This integrator accepts the following parameters in the initializer or 
set_options method of the dae class:

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
- nsteps : int
  Maximum number of (internally defined) steps allowed during one
  call to the solver.
- first_step : float
  Set initial stepsize. DAE solver can suffer on the first step, set this
  to circumvent this.
- max_step : float
  Limits for the step sizes used by the integrator. This overrides the
  internal value
- order : int
  Maximum order used by the integrator, >=1,  <= 5 for BDF.
  5 is the default
- enforce_nonnegativity: bool
  Enforce the nonnegativity of Y during integration
  Note: best is to run code first with no condition
- compute_initcond: None or 'yprime0' or 'yode0'
  DDASPK may be able to compute the initial conditions if you do not know them
  precisely. 
  If yprime0, then y0 will be calculated
  If yode0, then the differential variables will be used to solve for the 
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
- algebraic_var: integer array of length the number of unknowns, indicating the 
  algebraic variables in y. Use -1 to indicate an algebraic variable, and +1 for
  a differential variable.
"""


integrator_info_lsodi = \
"""
lsodi
~~~~
Solver developed during the 1980s, this is the version from 1987 - Fortran

Integrator for linearly implicit systems of first-order odes.  lsodi
provides the same methods as vode (adams and bdf).  This integrator
accepts the following parameters in initializer or set_options method of the 
dae class:
           atol=float|seq
           rtol=float
           lband=None|int
           rband=None|int
           method='adams'|'bdf'
           with_jacobian=0|1
           nsteps = int
           (first|min|max)_step = float
           tcrit=None|float
           order = int        # <=12 for adams, <=5 for bdf
           compute_initcond = None|'yode0'
           adda = function, required, see below
           rfn = the residual function, required
           jacfn = None|jacobian function

Details: 
- compute_initcond: None or 'yode0'
  LSODI may be able to compute the initial conditions if you do not know them
  precisely and the problem is not algebraic at t=0. 
  If yode0, then the differential variables (y of the ode system at time 0) 
    will be used to solve for the derivatives of the differential variables. 
Source: http://www.netlib.org/ode/lsodi.f

Implicit integration requires two functions do be defined to integrate 
                    d y(t)[i]
           a(t,y) * --------- = g(t,y)[i]
                        dt
where a(t,y) is a square matrix.

res returns an (neq, ) array and is the residual which the user provides. It
calculates something like
        def res(t, y, s, ires):
            r(t,y,s)=g(t,y)-a(t,y)*s 
            return r

adda must modify the provided matrix p.  If banded storage is used,
ml and mu provide the upper and lower diagonals, see the lsodi.f
source for full documentation.  adda is passed to the dae set_integrator()
with the adda= keyword argument.
        def adda(t,y,ml,mu,p,[nrowp]):
            p += a(t,y)
            return p
Note: if your residual is a s - g, then adda must substract a !

An optional jacobian can be specified which provides the derivatves of
r(t,y,s)=g(t,y)-a(t,y)*s
        def jac(t, y, s, ml, mu, nrowp):
                       d r(t,y,s)[i]
            p_{i,j} = --------------
                          d y[j]
            return p
"""



integrator_info = \
"""
Available integrators
---------------------
""" \
+ integrator_info_ddaspk \
+ integrator_info_lsodi \
+ \
"""
ddaskr  
~~~~~~
Not included, starting hints:
                 http://osdir.com/ml/python.f2py.user/2005-07/msg00014.html

Modified Extended Backward Differentiation Formulae (MEBDF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Not included. Fortran codes: http://www.ma.ic.ac.uk/~jcash/IVP_software/readme.html
"""
__doc__ += integrator_info

#
# How to define a new dae solver:
# ===============================
#
# class mydaeint(DaeIntegratorBase):
#
#     runner = <dae function> or None
#
#     def __init__(self,...):                           # required
#         <initialize>
#
#     def reset(self,n,has_jac):                        # optional
#         # n - the size of the problem (number of equations)
#         # has_jac - whether user has supplied its own routine for Jacobian
#         <allocate memory,initialize further>
#
#     def run(self,res,jac,y0,yprime0,t0,t1): # required
#         # this method is called to integrate from t=t0 to t=t1
#         # with initial condition y0 yprime0. 
#         #res and jac are user-supplied functions
#         # that define the problem.
#         <calculate y1>
#         if <calculation was unsuccesful>:
#             self.success = 0
#         return t1,y1,y1prime
#
#     # In addition, one can define step() and run_relax() methods (they
#     # take the same arguments as run()) if the integrator can support
#     # these features (see IntegratorBase doc strings).
#
# if mydaeint.runner:
#     DaeIntegratorBase.integrator_classes.append(myodeint)

__all__ = ['dae']
__version__ = "$Id$"
__docformat__ = "restructuredtext en"

from numpy import asarray, array, zeros, sin, int32, isscalar, empty, alen
from copy import copy
import re, sys

class DaeBase(object):
    """ the interface which DAE solvers must implement"""
    
    integrator_classes = []

    def __init__(self, Rfn, **options):
        """
        Initialize the DAE Solver and it's default values

        Input:
            Rfn     - residual function
            options - additional options for initialization
        """
        raise NotImplementedError('all DAE solvers must implement this')

    def set_options(self, **options):
        """
        Set specific options for the solver.
        """
        raise NotImplementedError('all DAE solvers must implement this')

    
    def init_step(self, t0, y0, yp0, y_ic0_retn = None, yp_ic0_retn = None):
        """
        Initializes the solver and allocates memory.

        Input:
            t0     - initial time
            y0     - initial condition for y (can be list or numpy array)
            yp0    - initial condition for yp (can be list or numpy array)
            y_ic0  - (optional) returns the calculated consistent initial condition for y
                     It MUST be a numpy array.
            yp_ic0 - (optional) returns the calculated consistent initial
                     condition for y derivated. It MUST be a numpy array.
        """
        raise NotImplementedError('all DAE solvers must implement this')

    def solve(self, tspan, y0,  yp0, hook_fn = None):
        """
        Runs the solver.
        
        Input:
            tspan - an list/array of times at which the computed value will be
                    returned. Must contain the start time.
            y0    - list/numpy array of initial values
            yp0   - list/numpy array of initial values of derivatives
            hook_fn  - if set, this function is evaluated after each successive 
                       internal) step. Input values: t, x, xdot, userdata. 
                       Output is 0 (success), otherwise computation is stopped 
                      and a return flag = ? is set. Values are stored in (see) t_err, y_err, yp_err
            
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
            If 'calc_initcond' option set, then solver returns instead of user 
            supplied y0, yp0 values as the starting values the values calculated 
            by the solver (i.e. consistent initial
            conditions. The starting time is then also the precomputed time.
        """
        raise NotImplementedError('all DAE solvers must implement this')

    def step(self, t, y_retn, yp_retn = None):
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
            t_out - time, where the solver stopped (when no error occured, t_out == t)
        """
        
##
##    runner = None            # runner is None => integrator is not available
##    success = None           # success==1 if integrator was called successfully
##    supports_run_relax = None
##    supports_step = None
##    integrator_classes = []
##    scalar = float
##    printinfo = False
##    
##    name = ''
##
##    def set_init_val(self, y, yprime, t, res, jac=None):
##        """Some backends might need initial values to set up themselves.
##           Note that the run routines also are passed init values!
##        """
##        self.res = res
##        self.jac = jac or (lambda :None)
##
##    def reset(self, n, has_jac):
##        """Prepare integrator for call: allocate memory, set flags, etc.
##        n - number of equations
##        has_jac - if user has supplied function for evaluating Jacobian.
##        """
##
##    def run(self, y0, yprime0, t0, t1):
##        """Integrate from t=t0 to t=t1 using y0 and yprime0as an initial 
##        condition.
##        Return 4-tuple (y1,y1prime,t1,istate) where y1,y1prime is the result 
##        and t=t1 defines the stoppage coordinate of the result.
##        """
##        raise NotImplementedError('all daeintegrators must define run(t0,t1,y0,yprime0,')
##        'res_params,jac_params)'
##
##    def step(self, y0, yprime0, t0, t1):
##        """Make one integration step and return (y1,t1)."""
##        raise NotImplementedError('%s does not support step() method' %\
##              (self.__class__.__name__))
##
##    def run_relax(self,y0,yprime0,t0,t1):
##        """Integrate from t=t0 to t>=t1 and return (y1,t)."""
##        raise NotImplementedError('%s does not support run_relax() method' %\
##              (self.__class__.__name__))
##    
##    def set_tcrit(self, tcrit=None):
##        """Change the tcrit value if possible in the solver without 
##            reinitializing the running solver
##        """
##        raise NotImplementedError

    #XXX: __str__ method for getting visual state of the integrator

class ddaspk(DaeBase):
    __doc__ += integrator_info_ddaspk
    
    try:
        import ddaspk as _ddaspk
    except ImportError:
        print(sys.exc_info()[1])
        _ddaspk = None
    _runner = getattr(_ddaspk,'ddaspk',None)
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

    def __init__(self,resfn, **options):
        default_values = {
            'rtol':1e-6,
            'atol':1e-12,
            'lband':None,
            'uband':None,
            'tcrit':None, 
            'order' : 5,
            'nsteps' : 500,
            'max_step' : 0.0, # corresponds to infinite
            'first_step' : 0.0, # determined by solver
            'enforce_nonnegativity':False, 
            'nonneg_type':None, 
            'compute_initcond':None,
            'constraint_init':False, 
            'constraint_type':None, 
            'algebraic_var':None, 
            'exclude_algvar_from_error':False, 
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
        self.nsteps = self.options['nsteps']
        self.max_step = self.options['max_step']
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
        elif re.match(self.options['compute_initcond'], r'yprime0', re.I): 
            self.compute_initcond = 2
        elif re.match(self.options['compute_initcond'], r'yode0', re.I): 
            self.compute_initcond = 1
        else: 
            raise ValueError('Unknown init cond calculation method %s' %(
                                            self.options['compute_initcond']))
        if self.compute_initcond == 1 and self.options['algebraic_var'] is None:
            raise ValueError('Give integer array indicating which are the '\
                              'algebraic variables, +1 for diffential var, '\
                              '-1 for algebraic var')
        self.algebraic_var = self.options['algebraic_var']
        self.excl_algvar_err = self.options['exclude_algvar_from_error']

        self.success = 1
    
    def init_step(self, t0, y0, yp0, y_ic0_retn = None, yp_ic0_retn = None):
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

##    def set_tcrit(self, tcrit=None):
##        """Change the tcrit value if possible in the solver without 
##            reinitializing the running solver
##        """
##        self.tcrit = tcrit
##        if self.tcrit is not None:
##            self.info[3] = 1
##            self.rwork[0] = self.tcrit
##        else:
##            self.info[3] = 0
##            self.rwork[0] = 0.

    def _reset(self, n, has_jac):
        # Calculate parameters for Fortran subroutine ddaspk.
        self.neq = n
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
        
        t_retn  = empty([alen(tspan), ], float)
        y_retn  = empty([alen(tspan), alen(y0)], float)
        yp_retn = empty([alen(tspan), alen(y0)], float)
        
        y_ic0_retn  = empty(alen(y0), float)
        yp_ic0_retn  = empty(alen(y0), float)
        tinit = self.init_step(tspan[0], y0, yp0, y_ic0_retn, yp_ic0_retn)
        
        t_retn[0] = tinit
        y_retn[0,:] = y0[:]
        yp_retn[0, :] = yp0[:]
        for ind, time in enumerate(tspan[1:]):
            if not self.success:
                break
            result = self.__run([2, 3], y_retn[ind], yp_retn[0], t_retn[ind], time)
            t_retn[ind+1] = result[2]
            y_retn[ind+1][:] = result[0][:]
            yp_retn[ind+1][:] = result[1][:]
        self.t = t_retn[-1]
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

class lsodi(DaeBase):
    __doc__ += integrator_info_lsodi

    try:
        import lsodi as _lsodi
    except ImportError:
        print(sys.exc_info()[1])
        _lsodi = None
    runner = getattr(_lsodi,'lsodi',None)
    _intdy = getattr(_lsodi,'intdy',None)
    
    name = 'lsodi'

    messages = {2 : 'lsodi was successful.',
               -1 : 'excess work done on this call (check all inputs).',
               -2 : 'excess accuracy requested (tolerances too small).',
               -3 : 'illegal input detected (see printed message).',
               -4 : 'repeated error test failures (check all inputs).',
               -5 : 'repeated convergence failures (perhaps bad jacobian'
                    ' supplied or wrong choice of tolerances).',
               -6 : 'error weight became zero during problem. (solution'
                    ' component i vanished, and atol or atol(i) = 0.).',
               -7 : 'cannot occur in casual use.',
               -8 : 'lsodi was unable to compute the initial dy/dt.  In'
                    ' casual use, this means a(t,y) is initially singular.'
                    '  Supply ydoti and use istate = 1 on the first call.'
               }
    supports_run_relax = 1
    supports_step = 1

    def __init__(self, resfn, **options):
        default_values = {
                'rtol': 1e-6,
                'atol': 1e-12,
                'lband': None,
                'uband': None,
                'tcrit': None, 
                'order': 0,
                'nsteps': 500,
                'max_step': 0.0, # corresponds to infinite
                'min_step': 0.0,
                'first_step': 0.0, # determined by solver
                'method': "adams", 
                'compute_initcond': None,
                'adda_func': None,
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
        self.order = self.options['order']
        self.nsteps = self.options['nsteps']
        self.max_step = self.options['max_step']
        self.min_step = self.options['min_step']
        self.first_step = self.options['first_step']
        if re.match(self.options['method'], r'adams', re.I): self.meth = 1
        elif re.match(self.options['method'], r'bdf', re.I): self.meth = 2
        else: raise ValueError('Unknown integration method %s'%(self.options['method']))
        if self.options['compute_initcond'] is None:
            self.compute_initcond = 0
        elif re.match(self.options['compute_initcond'], r'yode0', re.I):
            self.compute_initcond = 1
        else:
            raise ValueError('Unknown init cond calculation method %s' %(
                                            self.options['compute_initcond']))
                                            

        if self.options['adda_func'] is None:
            raise ValueError('adda_func is required for lsodi algorithm!')
        self.adda = self.options['adda_func']
        self.success = 1

    def init_step(self, t0, y0, yp0, y_ic0_retn = None, yp_ic0_retn = None):
        self._init_data()
        self.y0 = y0
        self.yp0 = yp0
        self.t = t0
        self._reset(len(y0), self.jac is not None)
        
        if self.compute_initcond:
            if self.first_step <= 0.:
                first_step = 1e-18
            else:
                first_step = self.first_step
            self.y, self.yp, t = self.__run(y0, yp0, t0, t0 + first_step)
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
        # Calculate parameters for Fortran subroutine lsodi.
        self.neq = n
        if has_jac:
            if self.mu is None and self.ml is None:
                miter = 1
            else:
                if self.mu is None: self.mu = 0
                if self.ml is None: self.ml = 0
                miter = 4
        else:
            if self.mu is None and self.ml is None:
                miter = 2
            else:
                if self.mu is None: self.mu = 0
                if self.ml is None: self.ml = 0
                miter = 5
        mf = 10*self.meth + miter
        if mf in [11,12]:
            lrw = 22 + 16*n + n**2
        elif mf in [14,15]:
            lrw = 22 + 17*n + (2*ml +mu)*neq
        elif mf in [21,22]:
            lrw = 22 + 9*n + n**2
        elif mf in [24,25]:
            lrw = 22 + 10*n + (2*ml +mu)*neq
        else:
            raise ValueError('Unexpected mf=%s'%(mf))
        liw = 20 + n
        rwork = zeros((lrw,), float)
        rwork[4] = self.first_step
        rwork[5] = self.max_step
        rwork[6] = self.min_step
        self.rwork = rwork
        iwork = zeros((liw,), int32)
        if self.ml is not None:
            iwork[0] = self.ml
        if self.mu is not None:
            iwork[1] = self.mu
        iwork[4] = self.order
        iwork[5] = self.nsteps
        iwork[6] = 2           # mxhnil
        self.iwork = iwork
        if isscalar(self.atol) :
            itol=1
        else:
            itol=2
        itask=1
        # don't internally compute ydot if it is provided
        if self.compute_initcond==0:
            istate=1
        else:
            istate=0
        if self.tcrit is not None:
            self.rwork[0]=self.tcrit
        iopt=1
        self.call_args = [itol,self.rtol,self.atol,itask,istate,iopt,
                            self.rwork,self.iwork,mf]
        #create storage
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
         the jacobian needed by lsodi
        """
        self.jac( t, y, yp, cj, self.tmp_jac)
        return self.tmp_jac

    def solve(self, tspan, y0, yp0, hook_fn = None):
        
        t_retn = empty([alen(tspan), ], float)
        y_retn = empty([alen(tspan), alen(y0)], float)
        yp_retn = empty([alen(tspan), alen(y0)], float)
        
        y_ic0_retn = empty(alen(y0), float)
        yp_ic0_retn = empty(alen(y0), float)
        tinit = self.init_step(tspan[0], y0, yp0, y_ic0_retn, yp_ic0_retn)
        
        t_retn[0] = tinit
        y_retn[0,:] = y0[:]
        yp_retn[0, :] = yp0[:]
        if self.tcrit is None:
            itask = 1
            self.call_args[3] = 1
        else:
            itask = self.call_args[3]
            self.call_args[3] = 4
        for ind, time in enumerate(tspan[1:]):
            result = self.__run(y_retn[ind], yp_retn[0], t_retn[ind], time)
            if not self.success:
                break
            t_retn[ind+1] = result[2]
            y_retn[ind+1][:] = result[0][:]
            yp_retn[ind+1][:] = result[1][:]
        self.t = t_retn[-1]
        return self.success, t_retn, y_retn, yp_retn, None, None, None

    def __run(self, *args):
        y1, y1prime_tmp, t, istate = self.runner(*((self._resFn, self.adda,
                                                 self._jacFn) + args[0:]
                                                 + tuple(self.call_args)) 
                                             )
        self.call_args[4] = istate
        y1prime = None
        if istate <0:
            print('lsodi:',self.messages.get(istate,'Unexpected istate=%s'%istate))
            if istate in [-1,-4,-5] :
                print('lsodi: present residual is', y1prime_tmp)
            self.success = 0
            
        if self.success:
            yh = self.rwork[20:]
            order = 1
            y1prime, iflag = self._intdy(t, order, yh, self.neq)
            if iflag<0:
                if iflag==-1: 
                    raise ValueError("order=%s invalid in call to intdy" \
                                                        %order)
                if iflag==-2: 
                    raise ValueError("t=%s invalid in call to intdy"%t)
        return y1, y1prime, t

    def step(self, t, y_retn, yp_retn = None):
        if not self.initialized:
            raise ValueError('Method ''init_step'' has to be called prior to the'
                    'first call of ''step'' method, or after changing options')
        if t > 0.0:
            if self.tcrit is None:
                itask = 1
                self.call_args[3] = 1
            else:
                itask = self.call_args[3]
                self.call_args[3] = 4
            self.y, self.yp, self.t = self.__run(self.y, self.yp, self.t, t)
            self.call_args[3] = itask
        else:
            itask = self.call_args[3]
            if self.tcrit is None:
                self.call_args[3] = 2
            else:
                self.call_args[3] = 5
            self.y, self.yp, self.t  = self.__run(self.y, self.yp, self.t, -t)
            self.call_args[3] = itask
        y_retn[:] = self.y[:]
        if yp_retn is not None:
            yp_retn[:] = self.yp[:]
        return self.call_args[4], self.t

if lsodi.runner:
    DaeBase.integrator_classes.append(lsodi)

try:
    from sundials import ida
    DaeBase.integrator_classes.append(ida.IDA)
    integrator_info_ida = """
    IDA solver from the SUNDIALS package. See info in 
    scikits.odes.sundials.ida.IDA class
    """
    __doc__ += integrator_info_ida
    integrator_info += integrator_info_ida
except ValueError as msg:
    print('Could not load IDA solver', msg)
except ImportError:
    print(sys.exc_info()[1])
    ida = None

#------------------------------------------------------------------------------
# User interface
#------------------------------------------------------------------------------

class dae(object):
    """\
A generic interface class to differential algebraic equation solvers.

See also
--------
odeint : an ODE integrator with a simpler interface based on lsoda from ODEPACK
ode : class around vode ODE integrator

Examples
--------
DAE arise in many applications of dynamical systems, as well as in 
discritisations of PDE (eg moving mesh combined with method of 
lines). 
As an easy example, consider the simple oscillator, which we write as 
G(y,y',t) = 0 instead of the normal ode, and solve as a DAE.

>>>from numpy import (arange, zeros, array, dot, sqrt, cos, sin, allclose)
>>>from scipy.integrate import dae
>>>class SimpleOscillator():
    #Free vibration of a simple oscillator::
    #    m \ddot{u} + k u = 0, u(0) = u_0, \dot{u}(0)=\dot{u}_0
    #Solution::
    #    u(t) = u_0*cos(sqrt(k/m)*t)+\dot{u}_0*sin(sqrt(k/m)*t)/sqrt(k/m)
    stop_t  = [2.09]
    u0      = 1.
    dotu0   = 0.1

    k = 4.0
    m = 1.0
    z0      = array([dotu0, u0], float)    #Free vibration of a simple oscillator::
    #    m \ddot{u} + k u = 0, u(0) = u_0, \dot{u}(0)=\dot{u}_0
    #Solution::
    #    u(t) = u_0*cos(sqrt(k/m)*t)+\dot{u}_0*sin(sqrt(k/m)*t)/sqrt(k/m)
    stop_t  = [2.09, 3.]
    u0      = 1.
    dotu0   = 0.1

    k = 4.0
    m = 1.0
    z0      = array([dotu0, u0], float)
    zprime0 = array([-k*u0, dotu0], float)

    def res(self, t, z, zp):
        tmp1 = zeros((2,2), float)
        tmp2 = zeros((2,2), float)
        tmp1[0,0] = self.m
        tmp1[1,1] = 1.
        tmp2[0,1] = self.k
        tmp2[1,0] = -1.
        return dot(tmp1, zp)+dot(tmp2, z)
    def solution(self, t):
        omega = sqrt(self.k / self.m)
        u = self.z0[1]*cos(omega*t)+self.z0[0]*sin(omega*t)/omega

>>> problem = SimpleOscillator()
>>> ig = dae(problem.res, None)
>>> ig.set_integrator('ddaspk')
>>> ig.set_initial_value(problem.z0, problem.zprime0,  t=0.0)
>>> z = [0]*len(problem.stop_t); zprime = [0]*len(problem.stop_t)
>>> i=0
>>> for time in problem.stop_t:
      z[i],  zprime[i] = ig.solve(time)
      i += 1
      assert ig.successful(), (problem,)
>>> for (time, zv) in zip(problem.stop_t, z):
      print 'calc', zv[1], ' ?? == ?? ', problem.solution(time)

"""

    __doc__ += integrator_info

    def __init__(self, integrator_name, eqsres, **options):
        """
        Define equation res = G(t,y,y') which can eg be G = f(y,t) - A y' when 
        solving A y' = f(y,t), 
        and where (optional) jac is the jacobian matrix of the nonlinear system
        see fortran source code), so d res/dy + scaling * d res/dy' or d res/dy
        depending on the backend

        Parameters
        ----------
        res : res(t, y, yprime, *res_args)
            Residual of the DAE. t is a scalar, y.shape == (n,), 
            yprime.shape == (n,)
            res_args is determined by the solver backend, set it as required by
            the backend you use, assume it to be unneeded
            res should return delta, status
            delta should be an array of the residuals, and status: 
              0 : continue
              -1: Illigal input, try again
              -2: Illigal input, stop
             It is not guaranteed that a solver takes this status into account 
        jac : jac(t, y, yprime, *jac_args)
            Jacobian of the rhs, typically 
                jac[i,j] = d res[i] / d y[j] + scaling *  d res[i] / d yprime[j]
            jac_args is determined by the solver backend, set it as required by
            the backend you use
        """
        
        integrator = find_dae_integrator(integrator_name)
        if integrator is None:
            raise ValueError('No integrator name match with %s or is not available.'\
                  %(repr(integrator_name)))
        else:
            self._integrator = integrator(eqsres, **options)

    def set_options(self, **options):
        self._integrator.set_options(**options)

    def init_step(self, t0, y0, yp0, y_ic0_retn = None, yp_ic0_retn = None):
        return self._integrator.init_step(t0, y0, yp0, y_ic0_retn, yp_ic0_retn)

    def solve(self, tspan, y0,  yp0, hook_fn = None):
        return self._integrator.solve(tspan, y0,  yp0, hook_fn)

    def step(self, t, y_retn, yp_retn = None):
        return self._integrator.step(t, y_retn, yp_retn)

#------------------------------------------------------------------------------
# DAE integrators
#------------------------------------------------------------------------------

def find_dae_integrator(name):
    for cl in DaeBase.integrator_classes:
        if re.match(name, cl.__name__, re.I) or re.match(name, cl.name, re.I):
            return cl
    return
