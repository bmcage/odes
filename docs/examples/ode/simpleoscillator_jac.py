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
import numpy as np

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

def jaceqn(t, x, fx, jac):
    jac[0,1] = 1
    jac[1,0] = -k/m

#instantiate the solver
from scikits.odes import ode
solver = ode('cvode', rhseqn, jacfn=jaceqn, )
#obtain solution at a required time
result = solver.solve([0., 10., 20.], initx)

print('\n   t        Solution          Exact')
print('------------------------------------')
#for t, u in zip(result[1], result[2]):
for t, u in zip(result.values.t, result.values.y):
    print('%4.2f %15.6g %15.6g' % (t, u[0], initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m)))

#continue the solver
#result = solver.solve([result[1][-1], result[1][-1]+1], result[2][-1])
result = solver.solve([result.values.t[-1], result.values.t[-1]+1, result.values.t[-1]+110], result.values.y[-1])
print('------------------------------------')
print('  ...continuation of the solution')
print('------------------------------------')

#for t, u in zip(result[1], result[2]):
for t, u in zip(result.values.t, result.values.y):
    print ('%4.2f %15.6g %15.6g' % (t, u[0], initx[0]*cos(sqrt(k/m)*t)+initx[1]*sin(sqrt(k/m)*t)/sqrt(k/m)))


from scikits.odes.sundials.cvode import CVODE, StatusEnum
from collections import namedtuple

def rootfn(t, x, g):
    g[0] = x[0]
    g[1] = x[1]
    g[2] = t - 10.
solver = CVODE(rhseqn, jacfn=jaceqn, old_api=False, one_step_compute=True,
               rootfn=rootfn, nr_rootfns=3)

next_tstop = 10.
solver.init_step(0., initx)
solver.set_options(tstop = next_tstop)
dense_t = []
dense_y = []
roots = []
Root = namedtuple("Root", ["index", "rootsfound"])
for cnt in range(1000):
    res = solver.step(1.)
    print(cnt, res.flag, res.values.t)
    dense_t.append(np.copy(res.values.t))
    dense_y.append(np.copy(res.values.y))
    match res.flag:
        case StatusEnum.ROOT_RETURN:
            rootsfound = solver.rootinfo()
            roots.append(Root(cnt, rootsfound))

        case StatusEnum.TSTOP_RETURN:
            break
t = np.array(dense_t)
y = np.array(dense_y)


    #if res.values.t > 9.99:
    #    break
    #print(cnt, res)
    #if res.values.y[0] <= 0.:
    #    break
