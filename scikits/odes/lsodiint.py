# Authors: B. Malengier 
"""
First-order DAE solver
"""

from __future__ import print_function

integrator_info = \
"""
lsodi
~~~~
Solver developed during the 1980s, this is the version from 1987 - Fortran

Integrator for linearly implicit systems of first-order odes.  lsodi
provides the same methods as vode (adams and bdf).  This integrator
accepts the following parameters in initializer or set_options method of the 
dae class:

- rfn : residual function, see below for signature. This option need not be
        set, as rfn will be set during initialization. If the residual function
        of initialization needs to be reset, this option can be used
- jacfn : jacobian function, see below for signature. Default is None.
- adda_func = function, required, see below
- atol=float|seq
- rtol=float
- lband=None|int
- rband=None|int
- method='adams'|'bdf'
- with_jacobian=0|1
- max_steps = int
- max_step_size = float
- (first|min|max)_step = float
- tcrit=None|float
- order = int        # <=12 for adams, <=5 for bdf
- compute_initcond = None|'yp0'

Details: 

Implicit integration requires two functions do be defined to integrate 
                    d y(t)[i]
           a(t,y) * --------- = g(t,y)[i]
                        dt
where a(t,y) is a square matrix.

- rfn computes the residual which the user provides.
  rfn has the signature: rfn(x, y, yprime, return_residual)
with 
    x       : independent variable, eg the time, float
    y       : array of n unknowns in x
    yprime  : dy/dx array of n unknowns in x, dimension = dim(y)
    return_residual: array that must be updated with the value of the residuals,
              so G(t,y,y').  The dimension is equal to dim(y)
    return value: Not needed. However, for use with other solvers, consider 
              returning 0 on success.
It calculates something like
        def res(t, y, yprime, r):
            r(t,y,s)=g(t,y)-a(t,y)*yprime

- adda_func must modify the provided matrix p and is a required option.  If banded storage is used,
ml and mu provide the upper and lower diagonals, see the lsodi.f
source for full documentation.  adda_func is passed to the dae via set_options
with the adda_func keyword argument. Schematically, it is
        def adda(t, y, ml, mu, p, nrowp):
            p += a(t,y)
            return p
Note: if your residual is a yprime - g, then adda must substract a, see def of
rfn !

- jacfn is not a supported option for lsodi.

- compute_initcond: None or 'yp0'
  LSODI may be able to compute the initial conditions if you do not know them
  precisely and the problem is not algebraic at t=0. 
  If yp0, then the differential variables (y of the ode system at time 0) 
    will be used to solve for the derivatives of the differential variables,
    so yp0 will be calculated. 
Source: http://www.netlib.org/ode/lsodi.f

"""

__doc__ += integrator_info

__all__ = ['lsodi']
__version__ = "$Id$"
__docformat__ = "restructuredtext en"

from numpy import asarray, array, zeros, sin, int32, isscalar, empty, alen
from copy import copy
from .dae import DaeBase
import re, sys

class lsodi(DaeBase):
    __doc__ += integrator_info

    try:
        from .lsodi import lsodi as _runner
        from .lsodi import intdy as _intdy
    except ImportError:
        print(sys.exc_info()[1])
        _runner = None
        _intdy = None

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
                'max_steps': 500,
                'max_step_size': 0.0, #corresponds to infinite
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
        if self.jac is not None:
            raise ValueError('Using provided Jacobian is not supported by lsodi')
        self.res = self.options['rfn']

        self.tcrit = self.options['tcrit']
        self.order = self.options['order']
        self.nsteps = self.options['max_steps']
        self.max_step = self.options['max_step_size']
        self.min_step = self.options['min_step']
        self.first_step = self.options['first_step']
        if re.match(self.options['method'], r'adams', re.I): self.meth = 1
        elif re.match(self.options['method'], r'bdf', re.I): self.meth = 2
        else: raise ValueError('Unknown integration method %s'%(self.options['method']))
        if self.options['compute_initcond'] is None:
            self.compute_initcond = 0
        elif re.match(self.options['compute_initcond'], r'yp0', re.I):
            self.compute_initcond = 1
        else:
            raise ValueError('Unknown init cond calculation method %s' %(
                                            self.options['compute_initcond']))
                                            

        if self.options['adda_func'] is None:
            raise ValueError('adda_func is required for lsodi algorithm!')
        self.adda = self.options['adda_func']
        self.success = 1

    def init_step(self, t0, y0, yp0, y_ic0_retn = None, yp_ic0_retn = None):
        """ See dae.DaeBase
        """
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
         the residual needed by lsodi
        """
        self.res(t, y, yp, self.tmp_res)
        return self.tmp_res

    def _jacFn(self, t, y, yp, cj):
        """Wrapper around the jacobian as defined by user, to create
         the jacobian needed by lsodi
        """
        ## Not supported for lsodi
        #self.jac( t, y, yp, cj, self.tmp_jac)
        return self.tmp_jac

    def solve(self, tspan, y0, yp0, hook_fn = None):
        """ See dae.DaeBase
        """
        
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
        intbreak = None
        for ind, time in enumerate(tspan[1:]):
            result = self.__run(y_retn[ind], yp_retn[ind], t_retn[ind], time)
            if not self.success:
                intbreak = ind+1
                break
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

    def __run(self, *args):
        y1, y1prime_tmp, t, istate = self._runner(*((self._resFn, self.adda,
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
        """ See dae.DaeBase
        """
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

if lsodi._runner:
    DaeBase.integrator_classes.append(lsodi)
