# Authors: B. Malengier based on ode.py

integrator_info_cvode = \
"""
odesCVODE
~~~

CVODE is the evolution of the VODE algorithm, see 
https://computation.llnl.gov/casc/sundials/main.html
To deliver CVODE we use pysundials
http://pysundials.sourceforge.net/

This code solves a system of differential/algebraic equations of the form 
y' = f(t,y) , using a combination of Backward Differentiation Formula 
(BDF) methods and a choice of two linear system solution methods: direct 
(dense or band) or Krylov (iterative). 
Krylov is not supported from within scikits.odes. 

Source: https://computation.llnl.gov/casc/sundials/main.html
        http://pysundials.sourceforge.net/

On construction the function calculating the rhs (f) must be given and 
optionally also the function calculating the jacobian (jac). 
f has the signature: f(x, y)
with 
    x : independent variable, eg the time, float
    y : array of n unknowns in x

return value should be an array with the result of the rhs computation

Jac has the signature jac(x, y, out, f) with
    x : independent variable, eg the time, float
    y : array of n unknowns in x where  jacobian=df_i/dy_j must be calculated
    out: the matrix (dense or bounded sundials matrix) that is the result of the
        computation
    f : computed value of f(x,y) that can be used in the determination of out
out should be a nxn shaped matrix in dense or banded style as per the
definition of lband/uband below. Jac is optional. 
Note that Jac is defined as df(i)/dy(j) 

This integrator accepts the following parameters in set_integrator()
method of the ode class:

- atol : float or sequence of length i
  absolute tolerance for solution
- rtol : float
  relative tolerance for solution
- method: adams or bdf. 
    This is the type of linear multistep method to be used.
    adams = Adams-Moulton is recommended for non-stiff problems.
    bdf = Backward Differentiation Formula is recommended for stiff problems.
- itertype: functional or newton
    specifies whether functional or newton iteration will be used.
    functional = does not require linear algebra
    newton = requires the solution of linear systems. Requires the 
            specification of a CVODE linear solver. (Recommended for stiff 
            problems).
- lband : None or int
- uband : None or int
  Jacobian band width, jac[i,j] != 0 for i-lband <= j <= i+uband.
  A banded algorithm will be used instead of a dense jacobian
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
  internal value. Default 0.0 (no limit)
- min_step : float
  Sets the absolute minimum of step size allowed. Default is 0.0 (no limit)
- order : int
  Maximum order used by the integrator, >=1,  <= 5 for BDF.
  5 is the 
- out : bool
  Indicates if the rhs function has signature f(t, y, out) (True) or instead
  out = f(t,y) (False). Default is False. Consider using the f(t,y,out) 
  signature as it is faster (no intermediate arrays needed)
"""

__all__ = []
__version__ = "$Id: odes_cvode bmalengier $"
__docformat__ = "restructuredtext en"


try:
    from pysundials import cvode
    from pysundials import nvecserial
except:
    print "Warning: cvode solver not available, pysundials needed"
    raise ImportError

from numpy import isscalar
import re
import ctypes

from scipy.integrate.ode import IntegratorBase

class odesCVODE(IntegratorBase):
    supports_run_relax = 0
    supports_step = 1
    scalar = nvecserial.numpyrealtype
    
    printinfo = False
    
    name = 'cvode'

    def __init__(self,
                 rtol=1e-6,atol=1e-12,
                 method='adams',
                 itertype='functional',
                 lband=None,uband=None,
                 krylov=None,  # None or spgmr
                 tcrit=None, 
                 order = None, # 12 for adams, 5 for bdf
                 nsteps = 500,
                 max_step = 0.0, # corresponds to infinite
                 min_step = 0.0, # corresponds to no limit of minimum step
                 first_step = 0.0, # determined by solver
                 stability_limit_detect = False,
                 nonlin_convergence_coef = 0.1,
                 maxerrtestfail = 7,
                 maxnonliniters = 3,
                 maxconvfails = 10,
                 out = False
                 ):
        if not  isscalar(rtol) :
            raise ValueError,'rtol (%s) must be a scalar for CVODE'\
                        % (rtol)
        self.rtol = rtol
        if not isscalar(atol) : 
            self.atol = nvecserial.NVector(list(atol))
        else:
            self.atol = atol
        self.mu = uband
        self.ml = lband
        self.krylov = krylov
        if self.krylov is not None and (self.mu is not None or self.ml is not None):
            raise ValueError, 'Krylov solver '+ str(self.krylov) + ' given, '\
                    'uband and lband should be None'
        if not self.krylov in [None, 'spgmr']:
            raise ValueError, 'krylov parameter should be None or spgmr'
        self.krylovprecond = cvode.PREC_NONE  #no preconditiong
        self.maxkrylovdim = 0 #max krylov subspace dim, use default (5)
        self.order = order
        if method == 'adams':
            self.method = cvode.CV_ADAMS
            if self.order is None:
                self.order = 12
        elif method == 'bdf':
            self.method = cvode.CV_BDF
            if self.order is None:
                self.order = 5
            if self.order > 5 or self.order < 1:
                raise ValueError, 'bdf order '+str(self.order)+' should be >=1, <=5'
        else:
            raise ValueError, 'method should adams or bdf'
        
        if itertype == 'functional':
            self.itert = cvode.CV_FUNCTIONAL
        elif itertype == 'newton':
            self.itert = cvode.CV_NEWTON
        else:
            raise ValueError, 'itertype should be functional or newton'

        self.tcrit = tcrit
        self.nsteps = nsteps
        self.max_step = max_step
        self.min_step = min_step
        self.first_step = first_step
        self.stablimdet = stability_limit_detect
        self.nlinconvcoef = nonlin_convergence_coef
        self.maxerrtestfail = maxerrtestfail
        self.maxnonliniters = maxnonliniters
        self.maxconvfails = maxconvfails
        
        self.cvode_mem = None
        self.useoutval = out
        self.success = 1
        self.return_code = None

    def successmsg(self):
        """ A message that constructed based on return_code and if run was
        successfull or not
        """
        if self.return_code == None:
            return ' - no outputmsg - '
        if self.success:
            return ' integration successfull, ret code %i' % self.return_code
        else:
            return ' ERROR: ret code %i' % self.return_code

    def set_tcrit(self, tcrit=None):
        """Change the tcrit value if possible in the solver without 
            reinitializing the running solver
        """
        if tcrit is not None:
            self.tcrit = tcrit
            cvode.CVodeSetStopTime(self.cvode_mem.obj, self.tcrit)
        else:
            if self.tcrit is not None:
                raise ValueError, 'Cannot unset tcrit once set, take a large'\
                        ' tcrit value instead.'
            
    def set_init_val(self, y, t, rhs, jac=None):
        """CVODE stores these and on a run the internal values are used
        """
        self.t = t
        self.y = y
        self.rhs = rhs
        self.jac = jac

    def reset(self, n, has_jac):
        # create the memory for the solver
        if self.cvode_mem is not None:
            del self.cvode_mem
        self.cvode_mem = cvode.CVodeMemObj(cvode.CVodeCreate(self.method,
                                                           self.itert))
        
        #allocate internal memory
        if isscalar(self.atol):
            errtype = cvode.CV_SS
        else:
            errtype = cvode.CV_SV
        cvode.CVodeMalloc(self.cvode_mem.obj, self._rhsFn, self.t,
                      nvecserial.NVector(self.y), 
                      errtype, 
                      self.rtol, self.atol)
        
        
        # do the set functions for the user choices
        # 1. order of the solver
        cvode.CVodeSetMaxOrd(self.cvode_mem.obj, self.order)
        # 2. max number of steps
        cvode.CVodeSetMaxNumSteps(self.cvode_mem.obj, self.nsteps)
        # 3. maximum step taken
        if self.max_step != 0.:
            cvode.CVodeSetMaxStep(self.cvode_mem.obj, self.max_step)
        # 4. critical time step not to pass
        if self.tcrit:
            cvode.CVodeSetStopTime(self.cvode_mem.obj, self.tcrit)
        # 5. initial step taken
        if self.first_step != 0.0: 
            cvode.CVodeSetInitStep(self.cvode_mem.obj, self.first_step)
        # 5. minimum  step taken
        if self.min_step != 0.:
            cvode.CVodeSetMinStep(self.cvode_mem.obj, self.min_step)
        # 6. maximum error test failures
        cvode.CVodeSetMaxErrTestFails(self.cvode_mem.obj, self.maxerrtestfail)
        # 7. maximum nonlinear iterations
        cvode.CVodeSetMaxNonlinIters(self.cvode_mem.obj, self.maxnonliniters)
        # 8. maximum convergence failures
        cvode.CVodeSetMaxConvFails(self.cvode_mem.obj, self.maxconvfails)
        # 6. stability limit detection
        cvode.CVodeSetStabLimDet(self.cvode_mem.obj, self.stablimdet)
        # 7. nonlinear convergence coefficient
        cvode.CVodeSetNonlinConvCoef(self.cvode_mem.obj, self.nlinconvcoef)

        #Attach linear solver module (Krylov not supported now!)
        if self.krylov is not None:
            if self.krylov == 'spgmr':
                cvode.CVSpgmr(self.cvode_mem.obj, self.krylovprecond, 
                              self.maxkrylovdim)
        elif self.ml is None and self.mu is None:
                
            #dense jacobian
            cvode.CVDense(self.cvode_mem.obj, n)
            if has_jac:
                cvode.CVDenseSetJacFn(self.cvode_mem.obj, self._jacDenseFn, None)
        else:
            #band jacobian
            if self.ml is None or self.mu is None:
                raise ValueError, 'Give both uband and lband, or nothing'
            cvode.CVBand(self.cvode_mem.obj, n, self.mu, self.ml)
            if has_jac:
                cvode.CVBandSetJacFn(self.cvode_mem.obj, self._jacBandFn, None)

        self.__yret =  cvode.NVector([0]*n)
        self.success = 1

    def _rhsFn(self, tt, yy, ydot, *args):
        """Wrapper function around the user provided rhs function so as to
           create the correct call sequence. Needed for CVode:
             tt (realtype)   the current value of the independent variable
             yy (NVector)    the vector of current dependent values
             ydot (NVector)	undefined values, contents should be set to the new values of y
             f_data         user data
        """
        if self.useoutval:
            self.rhs(tt, yy.asarray(), ydot.asarray())
        else:
            out = ydot.asarray()
            out[:] = self.rhs(tt, yy.asarray() )[:]
        
        return 0
    
    def _jacDenseFn(self, Neq, JJ, tt, yy, fy, jdata, 
                    tempv1, tempv2, tempv3):
        """Wrapper function around the user provided jac function so as to
           create the correct call sequence.
           JJ is a pysundials dense matrix, access is via JJ[i][j]
           cvode calls if dense: 
            Neq (int)     the length of all NVector arguments
            JJ (BandMat)  the matrix that will be loaded with anpproximation of
                          the Jacobian Matrix J = (df_i/dy_j) at the point (t,y).
            tt (realtype) the current value of the independent variable
            yy (NVector)  the current value of the dependent variable vector
            fy (NVector)  f(t,y)
            jdata (c_void_p) pointer to user data set by CVDenseSetJacFunc
            tmp1 (NVector)	 preallocated temporary working space
            tmp2 (NVector)	 preallocated temporary working space
            tmp3 (NVector)	 preallocated temporary working space
           cvode calls if band:
            Neq (int)       the length of all NVector arguments
            mupper (int)    upper band width
            mlower (int)    lower band width
            J (BandMat)	    the matrix that will be loaded with anpproximation 
                            of the Jacobian Matrix J = (df_i/dy_j) at the point (t,y).
            tt (realtype)   the current value of the independent variable
            yy (NVector)    the current value of the dependent variable vector
            fy (NVector)    f(t,y)
            jdata (c_void_p) pointer to user data set by CVBandSetJacFunc
            tmp1 (NVector)	 preallocated temporary working space
            tmp2 (NVector)	 preallocated temporary working space
            tmp3 (NVector)	 preallocated temporary working space
        """
        self.jac(tt, yy, JJ, fy)
        
        return 0

    def _jacBandFn(self, Neq, mupper, mlower, JJ, tt, yy, fy, jdata, 
                    tempv1, tempv2, tempv3):
        """Wrapper function around the user provided jac function so as to
           create the correct call sequence.
           JJ is a pysundials dense matrix, access is via JJ[i][j]
           cvode calls if dense: 
            Neq (int)     the length of all NVector arguments
            JJ (BandMat)  the matrix that will be loaded with anpproximation of
                          the Jacobian Matrix J = (df_i/dy_j) at the point (t,y).
            tt (realtype) the current value of the independent variable
            yy (NVector)  the current value of the dependent variable vector
            fy (NVector)  f(t,y)
            jdata (c_void_p) pointer to user data set by CVDenseSetJacFunc
            tmp1 (NVector)	 preallocated temporary working space
            tmp2 (NVector)	 preallocated temporary working space
            tmp3 (NVector)	 preallocated temporary working space
           cvode calls if band:
            Neq (int)       the length of all NVector arguments
            mupper (int)    upper band width
            mlower (int)    lower band width
            J (BandMat)	    the matrix that will be loaded with anpproximation 
                            of the Jacobian Matrix J = (df_i/dy_j) at the point (t,y).
            tt (realtype)   the current value of the independent variable
            yy (NVector)    the current value of the dependent variable vector
            fy (NVector)    f(t,y)
            jdata (c_void_p) pointer to user data set by CVBandSetJacFunc
            tmp1 (NVector)	 preallocated temporary working space
            tmp2 (NVector)	 preallocated temporary working space
            tmp3 (NVector)	 preallocated temporary working space
        """
        self.jac(tt, yy, JJ, fy)
        
        return 0

    def _run(self, state, rhs, jac, y0, t0, t1, *args):
        tret = cvode.realtype(t0)
        
        try:
            self.return_code = cvode.CVode(self.cvode_mem.obj, t1, 
                      self.__yret, ctypes.byref(tret), state)
        except AssertionError, msg:
            print msg
            self.success = 0

        if self.return_code < 0:
            self.success = 0

        if self.printinfo:
            self.info()

        return self.__yret.asarray(),  tret.value

    def run(self, *args):
        state = cvode.CV_NORMAL
        if self.tcrit:
            state = cvode.CV_NORMAL_TSTOP
        return self._run(state, *args)

    def step(self,*args):
        state = cvode.CV_ONE_STEP
        if self.tcrit:
            state = cvode.CV_ONE_STEP_TSTOP
        return self._run(state, *args)
    
    def info(self):
        nst = cvode.CVGetNumSteps(self.cvode_mem.obj)
        nni = cvode.CVGetNumNonlinSolvIters(self.cvode_mem.obj)
        nre = cvode.CVGetNumResEvals(self.cvode_mem.obj)
        netf = cvode.CVGetNumErrTestFails(self.cvode_mem.obj)
        ncfn = cvode.CVGetNumNonlinSolvConvFails(self.cvode_mem.obj)
        if self.ml is None and self.mu is None:
            nje = cvode.CVDenseGetNumJacEvals(self.cvode_mem.obj)
            nreLS = cvode.CVDenseGetNumResEvals(self.cvode_mem.obj)
        else:
            nje = cvode.CVBandGetNumJacEvals(self.cvode_mem.obj)
            nreLS = cvode.CVBandGetNumResEvals(self.cvode_mem.obj)

        print "-----------------------------------------------------------"
        print "Solve statistics: \n"
        print "Number of steps										= %ld"%(nst)
        print "Number of residual evaluations		 = %ld"%(nre+nreLS)
        print "Number of Jacobian evaluations		 = %ld"%(nje)
        print "Number of nonlinear iterations		 = %ld"%(nni)
        print "Number of error test failures			= %ld"%(netf)
        print "Number of nonlinear conv. failures = %ld"%(ncfn)
        

