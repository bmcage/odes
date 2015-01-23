# Authors: B. Malengier, C. Abert
"""
This example shows how a preconditioned iterative linear solver
is used to solve the Newton iterations arising from the solution
of the free vibration of a simple oscillator::
        m \ddot{u} + k u = 0, u(0) = u_0, \dot{u}(0) = \dot{u}_0
using the CVODE solver. The rhs function is given by \dot{u}. The
preconditioning is implemented in prec_solvefn which solves the
system P = I - gamma * J, where J is the Jacobian of the rhs.
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

#define function that solves the preconditioning system
def prec_solvefn(t, y, r, z, gamma, delta, lr):
    """Solve the preconditioning system I - gamma * J except
    for a constant factor."""
    z[0] =               r[0] + gamma * r[1]
    z[1] = - gamma*k/m * r[0] +         r[1]

#define function for the right-hand-side equations which has specific signature
def rhseqn(t, x, xdot):
    """ we create rhs equations for the problem"""
    xdot[0] = x[1]
    xdot[1] = - k/m * x[0]
    
#instantiate the solver using a left-preconditioned BiCGStab as linear solver
from scikits.odes import ode
solver = ode('cvode', rhseqn, linsolver='spbcg', precond_type='left', prec_solvefn = prec_solvefn)
#obtain solution at a required time
result = solver.solve([0., 1., 2.], initx)

print('\n   t        Solution          Exact')
print('------------------------------------')
for t, u in zip(result[1], result[2]):
    print('%4.2f %15.6g %15.6g' % (t, u[0], initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m)))

#continue the solver
result = solver.solve([result[1][-1], result[1][-1]+1], result[2][-1])
print('------------------------------------')
print('  ...continuation of the solution')
print('------------------------------------')

for t, u in zip(result[1], result[2]):
    print ('%4.2f %15.6g %15.6g' % (t, u[0], initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m)))
