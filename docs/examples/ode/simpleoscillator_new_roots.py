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
from numpy import asarray, cos, sin, sqrt, pi

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

def rootfn(t, x, g, userdata):
    g[0] = 0 if abs(t - pi/2) < 1e-3 else 1
def norootfn(t, x, g, userdata):
    g[0] = 1
#instantiate the solver
from scikits.odes import ode
solver = ode('cvode', rhseqn, rootfn=rootfn, nr_rootfns=1, old_api=False) # force new api
#obtain solution at a required time
soln = solver.solve([0., 1., 2.], initx)
result = soln.values

print('\n   t        Solution          Exact')
print('------------------------------------')
for t, u in zip(result.t, result.y):
    print('%4.2f %15.6g %15.6g' % (t, u[0], initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m)))


print('------------------------------------')
print('Root found at {:.6g}'.format(soln.roots.t[0]))

solver.set_options(rootfn=norootfn, nr_rootfns=1)

#continue the solver
result = solver.solve([soln.roots.t[0], soln.roots.t[0] + 1], soln.roots.y[0]).values
print('------------------------------------')
print('  ...continuation of the solution')
print('------------------------------------')

for t, u in zip(result.t, result.y):
    print ('%4.2f %15.6g %15.6g' % (t, u[0], initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m)))
