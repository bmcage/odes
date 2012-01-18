# Authors: B. Malengier 
"""
This example shows the most simple way of using a solver. 
We solve free vibration of a simple oscillator::
        m \ddot{u} + k u = 0, u(0) = u_0, \dot{u}(0) = \dot{u}_0
using the LSODI solver, which means we use residuals.
Solution::
        u(t) = u_0*cos(sqrt(k/m)*t)+\dot{u}_0*sin(sqrt(k/m)*t)/sqrt(k/m)
    
"""
from __future__ import print_function, division
from numpy import cos, sin, sqrt, empty

#data
k = 4.0
m = 1.0
#initial data on t=0, x[0] = u, x[1] = \dot{u}, xp = \dot{x}
initx = [1, 0.1]
initxp = [initx[1], -k/m*initx[0]]

#define function for the residual equations which has specific signature
def reseqn(t, x, xdot):
    """ we create residual equations for the problem
    This is = rhs(t,y)  -  a(t,y) * dy/dt
    """
    result = empty(2, float)
    result[0] = -m*xdot[1] - k*x[0]
    result[1] = -xdot[0] + x[1]
    return result

def adda(t, y, ml, mu, p, nrowp):
    """ adda function for lsodi. matrix a is the matrix in the 
    problem formulation you multiply wity dy/dt, and this method must
    add it with p
    """
    p[0,1] += m
    p[1,0] += 1.0
    return p
    
from scikits.odes import dae
solver = dae('lsodi', reseqn, adda_func=adda, rtol=([1e-4, 1e-4]),
                atol=([1e-6, 1e-10]))
#obtain solution at a required time
result = solver.solve([0., 1., 2.], initx, initxp)

print('\n   t        Solution          Exact')
print('------------------------------------')
for t, u in zip(result[1], result[2]):
    print('%4.2f %15.6g %15.6g' % (t, u[0], initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m)))

#continue the solver
result = solver.solve([result[1][-1], result[1][-1]+1], result[2][-1], 
                      result[3][-1])
print ('Continuation of the solution')
print ('t - Solution - Exact')
for t, u in zip(result[1], result[2]):
    print ('%4.2f %15.6g %15.6g' % (t, u[0], initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m)))
