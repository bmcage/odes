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
userdata = dict(
k = 4.0,
m = 1.0,
t1 = 10.,
rhs_calls = 0,
jac_calls = 0,
)
#initial data on t=0, x[0] = u, x[1] = \dot{u}, xp = \dot{x}
initx = [1, 0.1]

#define function for the right-hand-side equations which has specific signature
def rhseqn(t, x, xdot, my_user_data):
    """ we create rhs equations for the problem"""

    k = my_user_data['k']
    m = my_user_data['m']
    my_user_data['rhs_calls'] += 1
    xdot[0] = x[1]
    xdot[1] = - k/m * x[0]

def jaceqn(t, x, fx, jac, my_user_data):#=None):
    my_user_data['jac_calls'] += 1
    if my_user_data is None:
        print("ERROR")
        return 0

    k = my_user_data['k']
    m = my_user_data['m']

    jac[0,1] = 1
    jac[1,0] = -k/m

#instantiate the solver
if False:
    from scikits.odes.sundials.cvode import CVODE, StatusEnum, CV_WrapJacRhsFunction
    SolverClass = CVODE
else:
    from scikits.odes.sundials.cvode import CV_WrapJacRhsFunction
    from scikits.odes.sundials.cvodes import CVODES, StatusEnum
    SolverClass = CVODES

from collections import namedtuple

def rootfn(t, x, g, my_user_data):
    t1 = my_user_data['t1']
    g[0] = x[0]
    g[1] = x[1]
    g[2] = t - t1

solver = SolverClass(
    rhseqn, user_data=userdata,# jacfn=jaceqn,
    old_api=False, one_step_compute=True,
               rootfn=rootfn, nr_rootfns=3, )

next_tstop = 10.
#solver.init_step(0., initx)
solver.set_options(tstop = next_tstop)
dense_t = []
dense_y = []
roots = []
Root = namedtuple("Root", ["index", "rootsfound"])
print("starting loop")
#print(solver.get_info(),)
res = solver.solve([0, 10.0], initx)
print("completed:",res)
#print(solver.get_info(), solver.num_chk_pts, solver.options["rfn"])
