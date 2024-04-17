# Authors: B. Malengier
"""
This example shows the most simple way of using a solver.
We solve free vibration of a simple oscillator::
        m \ddot{u} + k u = 0, u(0) = u_0, \dot{u}(0) = \dot{u}_0
using the CVODE solver, which means we use a rhs function of \dot{u}.
Solution::
        u(t) = u_0*cos(sqrt(k/m)*t)+\dot{u}_0*sin(sqrt(k/m)*t)/sqrt(k/m)

"""
from __future__ import print_function
from numpy import asarray, cos, sin, sqrt

#data
k = 4.0
m = 1.0
#initial data on t=0, x[0] = u, x[1] = \dot{u}, xp = \dot{x}
initx = [1, 0.1]

#define function for the right-hand-side equations which has specific signature
def rhseqn(t, x, xdot):
    """ we create rhs equations for the problem"""
    xdot[0] = x[1]
    xdot[1] = - k/m * x[0]

#instantiate the solver
from scikits.odes import ode
solver = ode('cvode', rhseqn, tstop=1.5, old_api=False) # force new api
#obtain solution at a required time
soln = solver.solve([0., 1., 2.], initx)
result = soln.values
tstop = soln.tstop

print('\n   t        Solution          Exact')
print('------------------------------------')
for t, u in zip(result.t, result.y):
    print('%4.2f %15.6g %15.6g' % (t, u[0], initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m)))

print('------------------------------------')
print('Solution stopped at {0} with {1[0]:.4g}'.format(tstop.t[0], tstop.y[0]))

solver.set_options(tstop=5)
#continue the solver
result = solver.solve([tstop.t[0], tstop.t[0]+1], tstop.y[0]).values
print('------------------------------------')
print('  ...continuation of the solution')
print('------------------------------------')

for t, u in zip(result[0], result[1]):
    print ('%4.2f %15.6g %15.6g' % (t, u[0], initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m)))
