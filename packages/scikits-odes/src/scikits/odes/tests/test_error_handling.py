# Authors: B. Malengier, russel (scipy trac)
"""
Tests for differential algebraic equation solvers.
Here we test onroot and ontstop
"""
import numpy as np

from scikits.odes import ode
from scikits.odes.sundials.cvode import StatusEnum
from scikits.odes.sundials.common_defs import DTYPE

#data
g  = 9.81    # gravitational constant

Y0 = 1000.0  # Initial height
Y1 = 10.0    # Bottom height - when reached, changes to Y0 (teleport)
T1 = 10.0    # stop time
v0 = 0.0     # Initial speed

#initial data at t=0, y[0] = Y, y[1] = \dot{Y}
y0 = [Y0, v0]
t_end1 =  10.0 # Time of free fall for experiments 1,2
t_end2 = 100.0 # Time of free fall for experiments 3,4

atol = 1e-4
rtol = 1e-4


def rhs_fn_except(t, y, ydot):
    """ rhs equations for the problem """
    if t > 5:
        raise Exception("We can't go above t = 10")
    ydot[0] = y[1]
    ydot[1] = -g

def test_cvode_rhs_exception():
    #test calling sequence. End is reached before root is found
    tspan = np.arange(0, t_end1 + 1, 1.0, DTYPE)
    solver = ode('cvode', rhs_fn_except, old_api=False)
    soln = solver.solve(tspan, y0)
