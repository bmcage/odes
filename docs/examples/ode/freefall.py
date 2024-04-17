# -*- coding: utf-8 -*-
# Authors: P. Kišon
"""
This example shows the most simple way of using a solver.
We solve free falling of an object from height Y0 with teleportation
happening when the object reaches height Y1, Y1 < Y0. The object
is then teleported back to height Y0
We use the CVODE solver to demonstrate the 'onroot' option and
the reinit_IC() method of the CVODE solver.
"""
from __future__ import print_function
import numpy as np
from scikits.odes import ode

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
# On the other hand experiments 1 and 3 don't use the 'onroot',
# experiments 2 and 4 do and compute until the time t_end is reached
# (function onroot_va()). Experiment 5 stops after the first interruption
# (function onroot_vb()) occurs, whereas experiment 6 stops after the
# first interruption  at time t>28 (s) (function onroot_vc()).
# Otherwise all experiments are the same.

def rhs_fn(t, y, ydot):
    """ rhs equations for the problem """
    ydot[0] = y[1]
    ydot[1] = -g

def root_fn(t, y, out):
    """ root function to check the object reached height Y1 """
    out[0] = Y1 - y[0]
    return 0

def root_fn2(t, y, out):
    """ root function to check the object reached height Y1 """
    out[0] = Y1 - y[0]
    out[1] = (t-10)*(t-20)*(t-30)
    return 0

def root_fn3(t, y, out):
    """ root function to check the object reached time 10 """
    out[0] = (t - 10)*(t-20)*(t-30)
    return 0

def onroot_va(t, y, solver):
    """
    onroot function to reset the solver back at the start, but keep the current
    velocity
    """
    # Teleport the object back to height Y0, but retain its speed
    solver.reinit_IC(t, [Y0, y[1]])

    return 0

def onroot_vb(t, y, solver):
    """
    onroot function to stop solver when root is found
    """
    return 1

def onroot_vc(t, y, solver):
    """
    onroot function to reset the solver back at the start, but keep the current
    velocity as long as the time is less than a given amount
    """
    if t > 28: # we have found 4 interruption points, so we stop
        return 1
    solver.reinit_IC(t, [Y0, y[1]])

    return 0

def onroot_vd(t, y, solver):
    """
    onroot function to just continue if time <28
    """
    if t > 28:
        return 1

    return 0


def print_results(experiment_no, result, require_no_roots=False):
    ts, ys = result.values.t, result.values.y
    # Print computed values at tspan
    print('\n Experiment number: ', experiment_no)
    print('--------------------------------------')
    print('    t             Y               v')
    print('--------------------------------------')

    for (t, y, v) in zip(ts, ys[:, 0], ys[:, 1]):
        print('{:6.1f} {:15.4f} {:15.4f}'.format(t, y, v))


    t_roots, y_roots = result.roots.t, result.roots.y
    if not require_no_roots:
        # Print interruption points
        print('\n t_roots     y_roots        v_roots')
        print('--------------------------------------')
        if (t_roots is None) and (y_roots is None):
            print('{!s:6} {!s:15} {!s:15}'.format(t_roots, y_roots, y_roots))
        elif (t_roots is not None) and (y_roots is not None):
            if np.isscalar(t_roots):
                print('{:6.1f} {:15.4f} {:15.2f}'.format(t_roots, y_roots[0], y_roots[1]))
            else:
                for (t, y, v) in zip(t_roots, y_roots[:, 0], y_roots[:, 1]):
                    print('{:6.6f} {:15.4f} {:15.4f}'.format(t, y, v))
        else:
            print('Error: one of (t_roots, y_roots) is None while the other not.')
    else:
        print('Computation failed.')

# Set tspan to end at t_end1
tspan = np.arange(0, t_end1 + 1, 1.0, np.float)
#
# 1. Solve the problem without onroot function i.e. compute until
#    the first root is found
#
solver = ode('cvode', rhs_fn, nr_rootfns=1, rootfn=root_fn, old_api=False)
print_results(1, solver.solve(tspan, y0))

#
# 2. Solve the problem with onroot function onroot_va, which resets the problem
# with the current velocity when a root is found. Note that we expect no roots.
#
solver = ode('cvode', rhs_fn, nr_rootfns=1, rootfn=root_fn, onroot=onroot_va, old_api=False)
print_results(2, solver.solve(tspan, y0), require_no_roots=True)

# Set tspan to end at t_end2
tspan = np.arange(0, t_end2 + 1, 1.0, np.float)
#
# 3. Solve the problem without interrupt function i.e. compute until
#    the first root is found
#
solver = ode('cvode', rhs_fn, nr_rootfns=1, rootfn=root_fn, old_api=False)
print_results(3, solver.solve(tspan, y0))

#
# 4. Solve the problem with onroot function onroot_va, which resets the problem
# with the current velocity when a root is found. Note that unlike 2 we expect
# no roots.
#
solver = ode('cvode', rhs_fn, nr_rootfns=1, rootfn=root_fn, onroot=onroot_va, old_api=False)
print_results(4, solver.solve(tspan, y0))

#
# 5. Solve the problem with onroot function onroot_vb, which behaves similarly
# to the default, which is to compute until a root is found.
#
solver = ode('cvode', rhs_fn, nr_rootfns=1, rootfn=root_fn, onroot=onroot_vb, old_api=False)
print_results(5, solver.solve(tspan, y0))

#
# 6. Solve the problem with onroot function onroot_vc, which resets the problem
# like onroot_va, except that it stops after a certain time.
#
solver = ode('cvode', rhs_fn, nr_rootfns=1, rootfn=root_fn, onroot=onroot_vc, old_api=False)
print_results(6, solver.solve(tspan, y0))

#
# 7. 2 root functions, one with time
#
solver = ode('cvode', rhs_fn, nr_rootfns=2, rootfn=root_fn2, onroot=onroot_vc, old_api=False)
print_results(7, solver.solve(tspan, y0))

#
# 8. root functions with time, with root at end time
#
tspan = np.arange(0, 30 + 1, 1.0, np.float)
solver = ode('cvode', rhs_fn, nr_rootfns=1, rootfn=root_fn3, onroot=onroot_vc, old_api=False)
print_results(8, solver.solve(tspan, y0))
