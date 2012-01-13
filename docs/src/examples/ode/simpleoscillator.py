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
def rhseqn(t, x, xdot, result):
    """ we create rhs equations for the problem"""
    result[0] = - k/m * x[0]
    result[1] = x[1]
    
#instantiate the solver
from scikits.odes.sundials import cvode
solver = cvode.CVODE(rhseqn)
#obtain solution at a required time
result = solver.solve([0., 1., 2.], initx)

print ('t - Solution - Exact')
for t, u in zip(result[1], result[2]):
    print (t, u[0], '    - ', initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m))

#continue the solver
result = solver.solve([result[1][-1], result[1][-1]+1], result[2][-1], 
                      result[3][-1])
print ('Continuation of the solution')
print ('t - Solution - Exact')
for t, u in zip(result[1], result[2]):
    print (t, u[0], '    - ', initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m))