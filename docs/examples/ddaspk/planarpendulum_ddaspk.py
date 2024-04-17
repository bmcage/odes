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
u^2 + v^2 + \lambda l^2/m - g y = 0

This last expression will be used to find the solution to the planar 
pendulum problem. 

The algorithm first needs to find initial conditions for the derivatives,
then it solves the problme at hand. We take g=1, m=1

"""
from numpy import (arange, zeros, array, sin, empty)
from scikits.odes import dae
import pylab

class PlanarPendulum():
    
    stop_t  = arange(.0,15,1e-4)[1:]
    theta= 3.14/3 #starting angle
    x0=sin(theta)
    y0=-(1-x0**2)**.5

    g=1

    lambdaval = 0.1
    z0 =  array([x0, y0, 0., 0., lambdaval]) 
    zprime0 = array([0., 0., -lambdaval*x0, -lambdaval*y0-g, -g], float)
    
    def res(self, t, x, xdot, tmp):
        g=1
        tmp[0]=x[2]-xdot[0]
        tmp[1]=x[3]-xdot[1]
        tmp[2]=-xdot[2]-x[4]*x[0]
        tmp[3]=-xdot[3]-x[4]*x[1]-g
        #tmp[4]=x[0]*x[0]+x[1]*x[1]-1
        #tmp[4]=x[0]*x[2]+x[1]*x[3]
        tmp[4] = x[2]**2 + x[3]**2 \
                    - (x[0]**2 + x[1]**2)*x[4] - x[1] * PlanarPendulum.g

problem = PlanarPendulum()
z = empty((1+len(problem.stop_t), len(problem.z0)), float)
zprime = empty((1+len(problem.stop_t), len(problem.z0)), float)
algvar = -1

ig = dae('ddaspk', problem.res)
#first compute the correct initial condition from the values of z0
ig.set_options(algebraic_vars_idx=[4],
               compute_initcond='y0',
               first_step=1e-9,
               atol=1e-6, rtol=1e-6)
#initialize solver and obtain correct values of initial condition
tinit = ig.init_step(.0, problem.z0, problem.zprime0, z[0], zprime[0])

error = False
i = 1
for time in problem.stop_t:
        flag, tout = ig.step(time, z[i],  zprime[i] )
        i += 1
        #print 'sol at ', time, z[i]

        if flag < 1:
            error = True
            break

print('last sol', z[i-1], zprime[i-1])
test = empty(len(z[0]), float)
problem.res(problem.stop_t[i-2], z[i-1], zprime[i-1], test)
print('has residual: ', test)

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
