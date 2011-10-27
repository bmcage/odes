# Authors: B. Malengier 
"""
This example shows how to solve the planar pendulum in full coordinate space.
This results in a dae system with one algebraic equation. 

The problem is easily stated: a pendulum must move on a circle with radius 1, 
it has a mass m, and gravitational accelleration is g. 
The Lagragian is L = 1/2 m (u^2 + v^2) - m g y,
with constraint: x^2+y^2 = 1. 

Adding a Lagrange multiplier \lambda, we arrive at the Euler Lagrange 
differential equations for the problem:

\dot{x} = u
\dot{y} = v
\dot{u} = \lambda x/m
\dot{v} = \lambda y/m - g

and \lambda must be such that the constraint is satisfied:
x^2+y^2 = 1

DASPK cannot solve the above. Hence we derive a different constraint that
contains more of the unknowns, as well as \lambda. 

Derivation to time of the constraint gives a new constraint:
x u + y v =0

Derivating a second time to time gives us:
u^2 + v^2 + x \dot{u} + y \dot{v} = 0
which can be written with the known form of \dot{u}, \dot{v} as
u^2 + v^2 + \labmda l^2/m - g y = 0

This last expression will be used to find the solution to the planar 
pendulum problem. 

The algorithm first needs to find initial conditions for the derivatives,
then it solves the problme at hand. We take g=1, m=1

"""
from numpy import (arange, zeros, array, sin)
from common_defs import ResFunction
import numpy as np
import ida
#import pylab

class oscres(ResFunction):
    def evaluate(self, t, x, xdot, result):
        x=[1,2,3,4,5]
        xdot=[2,4,6,8,18]
        g=1
        result[0]=x[2]-xdot[0]
        result[1]=x[3]-xdot[1]
        result[2]=-xdot[2]-x[4]*x[0]
        result[3]=-xdot[3]-x[4]*x[1]-g
        #tmp[4]=x[0]*x[0]+x[1]*x[1]-1
        #tmp[4]=x[0]*x[2]+x[1]*x[3]
        result[4] = x[2]**2 + x[3]**2 \
                    - (x[0]**2 + x[1]**2)*x[4] - x[1] * g
        return 0
        
res=oscres()        

class SimpleOscillator():
    stop_t  = arange(.0,5,1e-2,dtype=np.float)
    theta= 3.14/3 #starting angle
    x0=sin(theta)
    y0=-(1-x0**2)**.5

    g=1

    lambdaval = 0.1
    z0 =  array([x0, y0, 0., 0., lambdaval], np.float) 
    zprime0 = array([0., 0., -lambdaval*x0, -lambdaval*y0-g, -g], np.float)
    
    

problem = SimpleOscillator()
z = [0]*(1+len(problem.stop_t)); zprime = [0]*(1+len(problem.stop_t))

print('Stepping in 1...')
solver=ida.IDA()
#raise NotImplemented('Wait a second... :D')
#ig = dae(problem.res, None)
#first compute the correct initial condition from the values of z0
#ig.set_integrator('odesIDA',algebraic_var=array([1,1,1,1,algvar]),
#                    compute_initcond='yode0',
#                    first_step=1e-9,
#                    atol=1e-6,rtol=1e-6)
solver.set_options(resfn=res,
                   compute_initcond='yode0',               
                   first_step=1e-9,
                   atol=1e-6,rtol=1e-6,
                   algvar=[4])
y1 = solver.run_solver(problem.stop_t, problem.z0, problem.zprime0)
#print("Dtype = ", problem.stop_t.dtype)
#y1 = solver.run_solver(problem.stop_t, problem.z0, problem.stop_t)
#ig.set_initial_value(problem.z0, problem.zprime0,  t=0.0)
print('Stepping out 1...')

raise NotImplemented('Wait to have it implemented...:D')

print('Stepping in 2...')
solver=ida.IDA()
solver.set_options(resfn=problem.res,
                   compute_initcond='yode0',               
                   first_step=1e-9,
                   atol=1e-6,rtol=1e-6,
                   algvar=[4])
solver.init_step(problem.stop_t[0], problem.z0, problem.zprime0)

error = False
for time in problem.stop_t[1:]:
    ct = solver.step(time)
    if ct < 0:
        error = True
        break

print('Stepping out 2...')

raise NotImplemented('Not this time...')

print('last sol', z[i-1], zprime[i-1])
print('has residual: ', problem.res(problem.stop_t[i-2], z[i-1], 
                                    zprime[i-1]))

nr = i
xt = [z[i][0] for i in range(nr)]
yt = [z[i][1] for i in range(nr)]
time = zeros(nr,float)
time[0] = 0.0
if error:
    time[1:]  = problem.stop_t[:nr-1] 
else:
    time[1:]  = problem.stop_t[:nr] 
    
pylab.figure(1)
pylab.subplot(211)
pylab.scatter(xt, yt)
pylab.axis('equal')
pylab.subplot(212)
pylab.plot(time, xt, 'b', time, yt, 'k')
pylab.axis()
pylab.show()
