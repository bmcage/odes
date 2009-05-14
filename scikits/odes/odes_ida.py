# Authors: B. Malengier based on ode.py

integrator_info = \
"""
odesIDA
~~~

IDA is the evolution of the daspk algorithm, see 
https://computation.llnl.gov/casc/sundials/main.html
To deliver IDA we use pysundials
http://pysundials.sourceforge.net/

This code solves a system of differential/algebraic equations of the form 
G(t,y,y') = 0 , using a combination of Backward Differentiation Formula 
(BDF) methods and a choice of two linear system solution methods: direct 
(dense or band) or Krylov (iterative). 
Krylov is not supported from within scikits.odes. 
In order to support it, a new interface should be created ddaspk_krylov, 
with a different signature, reflecting the changes needed.

Source: https://computation.llnl.gov/casc/sundials/main.html
        http://pysundials.sourceforge.net/

On construction the function calculating the residual (res) must be given and 
optionally also the function calculating the jacobian (jac). 
Res has the signature: res(x, y, yprime)
with 
    x : independent variable, eg the time, float
    y : array of n unknowns in x
    yprime : dy/dx array of n unknowns in x

return value should be an array with the result of the residual computation

Jac has the signature jac(x, y, yprime, cj) as res, however the return value 
should be a nxn shaped array in general or a banded shaped array as per the
definition of lband/uband belop. Jac is optional. 
Note that Jac is defined as dres(i)/dy(j) + cj*dres(i)/dyprime(j)

This integrator accepts the following parameters in set_integrator()
method of the ode class:

- atol : float or sequence of length i
  absolute tolerance for solution
- rtol : float or sequence of length i
  relative tolerance for solution
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
  internal value
- order : int
  Maximum order used by the integrator, >=1,  <= 5 for BDF.
  5 is the default
- compute_initcond: None or 'yprime0' or 'yode0'
  IDA may be able to compute the initial conditions if you do not know them
  precisely. 
  If yprime0, then y0 will be calculated
  If yode0, then the differential variables will be used to solve for the 
    algebraic variables and the derivative of the differential variables. 
    Which are the algebraic variables must be indicated with algebraic_var method
- compute_initcond_t0: float, default 0.01
  first value of t at which a solution will be requested (from IDASolve), Needed
  to determine direction of integration and a rough scale of variable t
- exclude_algvar_from_error: bool
  To determine solution, do not take the algebraic variables error control into 
  account. Default=False
- constraints: bool
  Enforce constraint checks on the solution Y
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

__all__ = []
__version__ = "$Id: odes_ida.py 2183 2009-04-23 07:50:12Z bmalengier $"
__docformat__ = "restructuredtext en"


try:
    from pysundials import ida
    from pysundials import nvecserial
except:
    print "Warning: ida solver not available, pysundials needed"
    raise ImportError

from numpy import isscalar
import re
import ctypes

import dae

class odesIDA(dae.DaeIntegratorBase):
    supports_run_relax = 0
    supports_step = 1
    scalar = nvecserial.numpyrealtype
    
    name = 'ida'

    def __init__(self,
                 rtol=1e-6,atol=1e-12,
                 lband=None,uband=None,
                 tcrit=None, 
                 order = 5,
                 nsteps = 500,
                 max_step = 0.0, # corresponds to infinite
                 first_step = 0.0, # determined by solver
                 compute_initcond=None,
                 compute_initcond_t0 = 0.01,
                 constraints=False, 
                 constraint_type=None, 
                 algebraic_var=None, 
                 exclude_algvar_from_error=False, 
                 ):
        if not  isscalar(rtol) :
            raise ValueError,'rtol (%s) must be a scalar for IDA'\
                        % (rtol)
        self.rtol = rtol
        if not isscalar(atol) : 
            self.atol = nvecserial.NVector(list(atol))
        else:
            self.atol = atol
        self.mu = uband
        self.ml = lband

        self.tcrit = tcrit
        if order > 5 or order < 1:
            raise ValueError, 'order should be >=1, <=5'
        self.order = order
        self.nsteps = nsteps
        self.max_step = max_step
        self.first_step = first_step
        if constraints and constraint_type is None:
            raise ValueError, 'Give type of contraint as '\
                              'an array (1:>=0, 2:>0, -1:<=0, -2:<0)'
        elif constraints:
            self.constraint_type = nvecserial.NVector(list(constraint_type))
        else:
            self.constraint_type = None
        if compute_initcond is None: self.compute_initcond = 0
        elif re.match(compute_initcond,r'yprime0',re.I): 
            self.compute_initcond = 2
        elif re.match(compute_initcond,r'yode0',re.I): self.compute_initcond = 1
        else: raise ValueError,'Unknown init cond calculation method %s' %(
                                                            compute_initcond)
        self.compute_initcond_t0 = compute_initcond_t0
        if self.compute_initcond == 1 and algebraic_var is None:
            raise ValueError, 'Give integer array indicating which are the '\
                              'algebraic variables, +1 for diffential var, '\
                              '-1 for algebraic var'
        #alg var in IDA is <=0 (officially 0) , differential var > 0 (off 1):
        if algebraic_var is not None:
            algebraic_var[algebraic_var<=0.] = 0.
            algebraic_var[algebraic_var>0.] = 1.
            self.algebraic_var = nvecserial.NVector(list(algebraic_var))
        else:
            self.algebraic_var = None
        self.excl_algvar_err = exclude_algvar_from_error
        self.ida_mem = None
        self.success = 1

    def set_tcrit(self, tcrit=None):
        """Change the tcrit value if possible in the solver without 
            reinitializing the running solver
        """
        
        if tcrit is not None:
            self.tcrit = tcrit
            ida.IDASetStopTime(self.ida_mem.obj, self.tcrit)
        else:
            if self.tcrit is not None:
                raise ValueError, 'Cannot unset tcrit once set, take a large'\
                        ' tcrit value instead.'
            
    def set_init_val(self, y, yprime, t, res, jac=None):
        """IDA stores these and on a run the internal values are used
        """
        self.t = t
        self.y = y
        self.yprime = yprime
        self.res = res
        self.jac = jac

    def reset(self,n,has_jac):
        # create the memory for the solver
        if self.ida_mem is not None:
            del self.ida_mem
            
        self.ida_mem = ida.IdaMemObj(ida.IDACreate())
        
        #allocate internal memory
        if isscalar(self.atol):
            errtype = ida.IDA_SS
        else:
            errtype = ida.IDA_SV
        ida.IDAMalloc(self.ida_mem.obj, self._resFn, self.t,
                      nvecserial.NVector(self.y), 
                      nvecserial.NVector(self.yprime), 
                      errtype, 
                      self.rtol, self.atol)
        
        
        # do the set functions for the user choices
        # 1. determine differential/algebraic variables
        if  self.algebraic_var == None:
            self.algebraic_var = nvecserial.NVector([1.]*n)
        ida.IDASetId(self.ida_mem.obj, self.algebraic_var)
        # 2. exclude alg var on error control
        if self.excl_algvar_err: 
            ida.IDASetSuppressAlg(self.ida_mem.obj, self.excl_algvar_err)
        # 3. set the contraints if given
        if self.constraint_type is not None:
            ida.IDASetConstraints(self.ida_mem.obj, self.constraint_type)
        
        # 4. order of the solver
        ida.IDASetMaxOrd(self.ida_mem.obj, self.order)
        # 5. max number of steps
        ida.IDASetMaxNumSteps(self.ida_mem.obj, self.nsteps)
        # 6. maximum step taken
        if self.max_step != 0.:
            ida.IDASetMaxStep(self.ida_mem.obj, self.max_step)
        # 7. critical time step not to pass
        if self.tcrit:
            ida.IDASetStopTime(self.ida_mem.obj, self.tcrit)
        # 8. initial step taken
        if self.first_step != 0.0: 
            ida.IDASetInitStep(self.ida_mem.obj, self.first_step)

        #Attach linear solver module (Krylov not supported now!)
        if self.ml is None and self.mu is None:
            #dense jacobian
            ida.IDADense(self.ida_mem.obj, n)
            if has_jac:
                ida.IDADenseSetJacFn(self.ida_mem.obj, self._jacDenseFn, None)
        else:
            #band jacobian
            if self.ml is None or self.mu is None:
                raise ValueError, 'Give both uband and lband, or nothing'
            ida.IDABand(self.ida_mem.obj, n, self.mu, self.ml)
            if has_jac:
                ida.IDABandSetJacFn(self.ida_mem.obj, self._jacBandFn, None)

        self.__yret =  ida.NVector([0]*n)
        self.__ypret =  ida.NVector([0]*n)
        self.success = 1

    def _resFn(self, t, yy, yp, resval, *args):
        """Wrapper function around the user provided res function so as to
           create the correct call sequence
        """
        out = resval.asarray()
        out[:] = self.res(t, yy.asarray(), yp.asarray() )[:]
        
        return 0
    
    def _jacDenseFn(self, Neq, tt, yy, yp, resvec, cj, jdata, JJ, 
                    tempv1, tempv2, tempv3):
        """Wrapper function around the user provided jac function so as to
           create the correct call sequence.
           JJ is a pysundials dense matrix, access is via JJ[i][j]
           ida calls if dense: 
                (long int Neq, realtype tt, N_Vector yy, N_Vector yp,
                         N_Vector rr, realtype c_j, void *jac_data, DenseMat Jac,
                         N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
           ida calls if band:
                (long int Neq, long int mupper, long int mlower,
                        realtype tt, N_Vector yy, N_Vector yp, N_Vector rr,
                        realtype c_j, void *jac_data, BandMat Jac,
                        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
        """
        self.jac(tt, yy, yp, cj, JJ)
        
        return 0

    def _jacBandFn(self, Neq, tt, yy, yp, resvec, cj, jdata, JJ, 
                    tempv1, tempv2, tempv3):
        """Wrapper function around the user provided jac function so as to
           create the correct call sequence.
           JJ is a pysundials band matrix, access is via JJ[i][j]
           ida calls if dense: 
                (long int Neq, realtype tt, N_Vector yy, N_Vector yp,
                         N_Vector rr, realtype c_j, void *jac_data, DenseMat Jac,
                         N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
           ida calls if band:
                (long int Neq, long int mupper, long int mlower,
                        realtype tt, N_Vector yy, N_Vector yp, N_Vector rr,
                        realtype c_j, void *jac_data, BandMat Jac,
                        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
        """
        self.jac(tt, yy, yp, cj, JJ)
        
        return 0

    def _run(self, state, y0, yprime0, t0, t1, *args):
        if self.compute_initcond: 
            #this run we compute the initial condition first
            if self.compute_initcond == 1:
                ida.IDACalcIC(self.ida_mem.obj, ida.IDA_YA_YDP_INIT, 
                                self.compute_initcond_t0)
            if self.compute_initcond == 2:
                ida.IDACalcIC(self.ida_mem.obj, ida.IDA_Y_INIT, 
                                self.compute_initcond_t0)
            self.compute_initcond = 0
            n = len(self.y)
            corry = ida.NVector([0]*n)
            corryp = ida.NVector([0]*n)
            ida.IDAGetConsistentIC(self.ida_mem.obj, corry, corryp)
            return corry, corryp, 0.
        
        tret = ida.realtype(t0)
        
        try:
            ida.IDASolve(self.ida_mem.obj, t1, ctypes.byref(tret), 
                     self.__yret, self.__ypret, state)
        except AssertionError, msg:
            print msg
            self.success = 0
            
        if self.printinfo:
            self.info()

        return self.__yret.asarray().copy(), self.__ypret.asarray().copy(),  \
                tret.value

    def run(self, *args):
        state = ida.IDA_NORMAL
        if self.tcrit:
            state = ida.IDA_NORMAL_TSTOP
        return self._run(state, *args)

    def step(self,*args):
        state = ida.IDA_ONE_STEP
        if self.tcrit:
            state = ida.IDA_ONE_STEP_TSTOP
        return self._run(state, *args)
    
    def info(self):
        nst = ida.IDAGetNumSteps(self.ida_mem.obj)
        nni = ida.IDAGetNumNonlinSolvIters(self.ida_mem.obj)
        nre = ida.IDAGetNumResEvals(self.ida_mem.obj)
        netf = ida.IDAGetNumErrTestFails(self.ida_mem.obj)
        ncfn = ida.IDAGetNumNonlinSolvConvFails(self.ida_mem.obj)
        nje = ida.IDABandGetNumJacEvals(self.ida_mem.obj)
        nreLS = ida.IDABandGetNumResEvals(self.ida_mem.obj)

        print "-----------------------------------------------------------"
        print "Solve statistics: \n"
        print "Number of steps										= %ld"%(nst)
        print "Number of residual evaluations		 = %ld"%(nre+nreLS)
        print "Number of Jacobian evaluations		 = %ld"%(nje)
        print "Number of nonlinear iterations		 = %ld"%(nni)
        print "Number of error test failures			= %ld"%(netf)
        print "Number of nonlinear conv. failures = %ld"%(ncfn)
        
