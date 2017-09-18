# Test different solvers from 'scipy.integrate.ode' and 'odes'
# with Van der Pol equation in stiff regime

# Optionally use cython for the RHS evaluation.
#Then you need to compile van_der_pol_fun.pyx, see
# instructions at the top of the file.

from __future__ import division
import pylab
import numpy as np
from numpy import array
from scipy.integrate import ode
from scikits.odes import ode as ode_odes
from scikits.odes.sundials.cvode import CV_RhsFunction

HAS_CYTHON_RHS = True
try:
    from van_der_pol_fun import CV_Rhs_van_der_pol_cy, van_der_pol_cy
except:
    HAS_CYTHON_RHS = False

import time

mu = 1000
y0 = np.array([0.5, 0.5])
Te = 500
tt = np.linspace(1,Te,200)

def van_der_pol(t, y):
    return np.array([y[1], mu*(1.0-y[0]**2)*y[1]-y[0]])

def van_der_pol_odes(t, y, ydot):
    """ we create rhs equations for the problem"""
    ydot[0] = y[1]
    ydot[1] = mu*(1.0-y[0]**2)*y[1]-y[0]

class CV_Rhs_van_der_pol(CV_RhsFunction):
    def evaluate(self, t, y, ydot, userdata):
        ydot[0] = y[1]
        ydot[1] = mu*(1.0-y[0]**2)*y[1]-y[0]
        return 0

r1a = ode(van_der_pol).set_integrator('dop853', nsteps=1000)
r1b = ode_odes('dop853', van_der_pol_odes, nsteps=1000)

r1a.set_initial_value(y0, t=tt[0])

r2 = ode(van_der_pol).set_integrator('vode', method='bdf', with_jacobian=True)
r2.set_initial_value(y0, t=tt[0])

r3 = ode(van_der_pol).set_integrator('lsoda')
r3.set_initial_value(y0, t=tt[0])

r4 = ode_odes('cvode', van_der_pol_odes, old_api=False)

r5 = ode_odes('cvode', CV_Rhs_van_der_pol(), old_api=False)

if HAS_CYTHON_RHS:
    r6 = ode_odes('cvode', CV_Rhs_van_der_pol_cy(), old_api=False)
    r7 = ode(van_der_pol_cy).set_integrator('lsoda')
    r7.set_initial_value(y0, t=tt[0])
    r8 = ode(van_der_pol_cy).set_integrator('vode', method='bdf', with_jacobian=True)
    r8.set_initial_value(y0, t=tt[0])

c1a = time.clock()
sol1a = [r1a.integrate(T) for T in tt[1:]]
print ('END dop853 orig, now dop853 via odes scikit')
c1b = time.clock()
sol1b = r1b.solve(tt, y0)
c1c = time.clock()
# restart solver from the error
print ('error reached, restart once ...')
tt2 = np.linspace(sol1b.errors.t,Te,134)
r1b.set_options(nsteps=100000)
sol1c = r1b.solve(tt2, sol1b.errors.y)
if sol1c.errors.t:
    print ('... again error. Stop. Starting other solvers now')

c2 = time.clock()
sol2 = [r2.integrate(T) for T in tt[1:]]

c3 = time.clock()
sol3 = [r3.integrate(T) for T in tt[1:]]

c4 = time.clock()
sol4 = r4.solve(tt, y0)
c5 = time.clock()

sol5 = r5.solve(tt, y0)
c6 = time.clock()

if HAS_CYTHON_RHS:
    sol6 = r6.solve(tt, y0)
    c7 = time.clock()
    sol7 = [r7.integrate(T) for T in tt[1:]]
    c8 = time.clock()
    sol8 = [r8.integrate(T) for T in tt[1:]]
    c9 = time.clock()

sol1a = array([[0.5, 0.5]] + sol1a)
sol2 = array([[0.5, 0.5]] + sol2)
sol3 = array([[0.5, 0.5]] + sol3)

print ("Time for dop853 orig:" + str(c1b-c1a))
print ("Time for dop853 odes:" + str(c1c-c1b + c2-c1c))
print ("Time for vode/BDF:   " + str(c3-c2))
print ("Time for lsoda:      " + str(c4-c3))
print ("Time for cvode/BDF:  " + str(c5-c4))
print ("Time for cvode/BDF - class:  " + str(c6-c5))

if HAS_CYTHON_RHS:
    sol7 = array([[0.5, 0.5]] + sol7)
    sol8 = array([[0.5, 0.5]] + sol8)
    print ("Time for cvode/BDF - cython: " + str(c7-c6))
    print ("Time for lsoda     - cython: " + str(c8-c7))
    print ("Time for vode/BDF  - cython: " + str(c9-c8))

##pylab.ion()
pylab.figure()
pylab.plot(tt, sol1a[:,0], label='dop853 orig')
pylab.plot(sol1b.values.t, sol1b.values.y[:,0], label='dop853')
if HAS_CYTHON_RHS:
    pylab.plot(tt, sol8[:,0], label='vode/BDF  - cython')
    pylab.plot(sol6.values.t, sol6.values.y[:,0], label='cvode/BDF - cython')
    pylab.plot(tt, sol7[:,0], label='lsoda - cython')
else:
    pylab.plot(tt, sol2[:,0], label='vode/BDF')
    pylab.plot(sol4.values.t, sol4.values.y[:,0], label='cvode/BDF')
    pylab.plot(tt, sol3[:,0], label='lsoda')
pylab.legend()
pylab.show()
##raw_input("press key to exit")
##pylab.ioff()