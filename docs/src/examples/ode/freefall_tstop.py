# -*- coding: utf-8 -*-
# Authors: P. Kišon
"""
This example shows the most simple way of using a solver.
We solve free falling of an object from height Y0 with teleportation
happening when the object reaches height Y1, Y1 < Y0. The object
is then teleported back to height Y0
We use the CVODE solver to demonstrate the 'ontstop' option and
the reinit_IC() method of the CVODE solver.
"""
from __future__ import print_function
import numpy as np
from scikits.odes.sundials import cvode

#data
g  = 9.81    # gravitational constant

Y0 = 1000.0  # Initial height
Y1 = 10.0    # Bottom height - when reached, changes to Y0 (teleport)
v0 = 0.0     # Initial speed

#initial data at t=0, y[0] = Y, y[1] = \dot{Y}
y0 = [Y0, v0]
t_end1 =  10.0 # Time of free fall for experiments 1,2
t_end2 = 100.0 # Time of free fall for experiments 3,4

# Experiments description:
#
# Experiments 1 and 2 have the same time of falling 't_end1', while
# experiments 3, 4, 5 and 6 have the time 't_end2'.
# On the other hand experiments 1 and 3 don't use the 'ontstop',
# experiments 2 and 4 do and compute until the time t_end is reached
# (function ontstop_va()). Experiment 5 stops after the first interruption
# (function ontstop_vb()) occurs, whereas experiment 6 stops after the
# first interruption  at time t>28 (s) (function ontstop_vc()).
# Otherwise all experiments are the same.

def rhs_fn(t, y, ydot):
    """ rhs equations for the problem """
    ydot[0] = y[1]
    ydot[1] = -g

def ontstop_va(t, y, solver):
    """ function to handle interrupt cause by finding the tstop """

    # Teleport the object back to height Y0, but retain its speed
    solver.reinit_IC(t, [Y0, y[1]])

    return 0

def ontstop_vb(t, y, solver):
    return 1

def ontstop_vc(t, y, solver):
    if t > 28: # we have found 4 interruption points, so we stop
        return 1

    solver.reinit_IC(t, [Y0, y[1]])

    return 0


def print_results(experiment_no, result, require_no_tstop=False):
    ts, ys = result.values.t, result.values.y
    # Print computed values at tspan
    print('\n Experiment number: ', experiment_no)
    print('--------------------------------------')
    print('    t             Y               v')
    print('--------------------------------------')

    for (t, y, v) in zip(ts, ys[:, 0], ys[:, 1]):
        print('%6.1f %15.4f %15.2f' % (t, y, v))


    t_tstop, y_tstop = result.tstop.t, result.tstop.y
    if not require_no_tstop:
        # Print interruption points
        print('\n t_tstop     y_tstop        v_tstop')
        print('--------------------------------------')
        if (t_tstop is None) and (y_tstop is None):
            print('%6s %15s %15s' % (t_tstop, y_tstop, y_tstop))
        elif (t_tstop is not None) and (y_tstop is not None):
            if np.isscalar(t_tstop):
                print('%6.1f %15.4f %15.2f' % (t_tstop, y_tstop[0], y_tstop[1]))
            else:
                for (t, y, v) in zip(t_tstop, y_tstop[:, 0], y_tstop[:, 1]):
                    print('%6.1f %15.4f %15.2f' % (t, y, v))
        else:
            print('Error: one of (t_tstop, y_tstop) is None while the other not.')
    else:
        print('Computation failed.')

# Set tspan to end at t_end1
tspan = np.arange(0, t_end1 + 1, 1.0, np.float)
#
# 1. Solve the problem without interrupt function i.e. compute until
#    the first tstop is found
#
solver = cvode.CVODE(rhs_fn, tstop=10, old_api=False)
print_results(1, solver.solve(tspan, y0))

#
# 2. Solve the problem with interrupt function i.e. compute until
#    the final time is reached
#
solver = cvode.CVODE(rhs_fn, tstop=10, ontstop=ontstop_va, old_api=False)
print_results(2, solver.solve(tspan, y0), require_no_tstop=True)

# Set tspan to end at t_end2
tspan = np.arange(0, t_end2 + 1, 1.0, np.float)
#
# 3. Solve the problem without interrupt function i.e. compute until
#    the first tstop is found
#
solver = cvode.CVODE(rhs_fn, tstop=10, old_api=False)
print_results(3, solver.solve(tspan, y0))

#
# 4. Solve the problem with interrupt function i.e. compute until
#    the final time is reached
#
solver = cvode.CVODE(rhs_fn, tstop=10, ontstop=ontstop_va, old_api=False)
print_results(4, solver.solve(tspan, y0))

#
# 5. Solve the problem without interrupt function i.e. compute until
#    the first tstop is found
#
solver = cvode.CVODE(rhs_fn, tstop=10, ontstop=ontstop_vb, old_api=False)
print_results(5, solver.solve(tspan, y0))

#
# 6. Solve the problem with interrupt function i.e. compute until
#    the final time is reached
#
solver = cvode.CVODE(rhs_fn, tstop=10, ontstop=ontstop_vc, old_api=False)
print_results(6, solver.solve(tspan, y0))
