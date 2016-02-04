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
solver = ode('cvode', rhseqn, old_api=True)
#obtain solution at a required time
result = solver.solve([0., 10., 20.], initx)

print ('\n sundials cvode old API')
print('\n   t        Solution          Exact')
print('------------------------------------')
for t, u in zip(result[1], result[2]):
    print('%4.2f %15.6g %15.6g' % (t, u[0], initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m)))

#continue the solver
result = solver.solve([result[1][-1], result[1][-1]+10, result[1][-1]+110], result[2][-1])
print('------------------------------------')
print('  ...continuation of the solution')
print('------------------------------------')

for t, u in zip(result[1], result[2]):
    print ('%4.2f %15.6g %15.6g' % (t, u[0], initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m)))


solver = ode('cvode', rhseqn, old_api=False)
#obtain solution at a required time
result = solver.solve([0., 10., 20.], initx)
print ('\n sundials cvode new API')
print('\n   t        Solution          Exact')
print('------------------------------------')
for t, u in zip(result.values.t, result.values.y):
    print('%4.2f %15.6g %15.6g' % (t, u[0], initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m)))

#continue the solver
result = solver.solve([result.values.t[-1], result.values.t[-1]+10, result.values.t[-1]+110], result.values.y[-1])
print('------------------------------------')
print('  ...continuation of the solution')
print('------------------------------------')

for t, u in zip(result.values.t, result.values.y):
    print ('%4.2f %15.6g %15.6g' % (t, u[0], initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m)))

if result.errors.t:
    print ('Error at time {}, message: {}'.format(result.errors.t, result.message))


from scipy.integrate import ode as scode
#define function for the right-hand-side equations which has specific signature
def scrhseqn(t, x):
    """ we create rhs equations for the problem"""
    return [ x[1],  - k/m * x[0]]
solver = scode(scrhseqn).set_integrator('vode', method='bdf')
solver.set_initial_value(initx,0)
#obtain solution at a required time
print ('\n scipy vode (fails some seconds earlier than cvode)')
print('\n   t        Solution          Exact')
print('------------------------------------')

while solver.successful() and solver.t < 30:
    solver.integrate(solver.t+10)
    print('%4.2f %15.6g %15.6g' % (solver.t, solver.y[0], initx[0]*cos(sqrt(k/m)*solver.t)+initx[1]*sin(sqrt(k/m)*solver.t)/sqrt(k/m)))

solver.integrate(solver.t+100)
print('%4.2f %15.6g %15.6g' % (solver.t, solver.y[0], initx[0]*cos(sqrt(k/m)*solver.t)+initx[1]*sin(sqrt(k/m)*solver.t)/sqrt(k/m)))