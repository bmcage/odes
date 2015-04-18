# Authors: P. Kišon
"""
This example shows the most simple way of using a solver.
We solve free falling of an object from height Y0 with teleportation
happening when the object reaches height Y1, Y1 < Y0. The object
is then teleported back to height Y0
We use the CVODE solver to demonstrate the 'interruptfn' option and
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
# On the other hand experiments 1 and 3 don't use the 'interruptfn',
# experiments 2 and 4 do and compute until the time t_end is reached
# (function interr_fn()). Experiment 5 stops after the first interruption
# (function interr_fn2()) occurs, whereas experiment 6 stops after the
# first interruption  at time t>28 (s) (function interr_fn3()).
# Otherwise all experiments are the same.

def rhs_fn(t, y, ydot):
    """ rhs equations for the problem """
    ydot[0] = y[1]
    ydot[1] = -g

def root_fn(t, y, out):
    """ root function to check the object reached height Y1 """
    #print('t = ', t, ', Y = ', y[0], ', v = ', y[1])
    #print('Y1=', Y1, x)
    out[0] = 10 - y[0]
    return 0

def interr_fn(flag, t, y, data):
    """ function to handle interrupt cause by finding the root """
    # Note: we know that flag can only 2 == Root found, so no need to check

    # Teleport the object back to height Y0, but retain its speed
    solver.reinit_IC(t, [Y0, y[1]])

    return 0

def interr_fn2(flag, t, y, data):
    return 1

def interr_fn3(flag, t, y, data):
    if t > 28: # we have found 4 interruption points, so we stop
        return 1

    solver.reinit_IC(t, [Y0, y[1]])

    return 0


def print_results(experiment_no, t, y, t_interr, y_interr):
    # Print computed values at tspan
    print('\n Experiment number: ', experiment_no)
    print('--------------------------------------')
    print('    t             Y               v')
    print('--------------------------------------')

    for (t, y, v) in zip(t_retn, y_retn[:, 0], y_retn[:, 1]):
        print('%6.1f %15.4f %15.2f' % (t, y, v))

    # Print interruption points
    print('\n t_interr     Y_interr        v_interr')
    print('--------------------------------------')

    if (t_interr is None) and (y_interr is None):
        print('%6s %15s %15s' % (t_interr, y_interr, y_interr))
    elif (t_interr is not None) and (y_interr is not None):
        if np.isscalar(t_interr):
            print('%6.1f %15.4f %15.2f' % (t_interr, y_interr[0], y_interr[1]))
        else:
            for (t, y, v) in zip(t_interr, y_interr[:, 0], y_interr[:, 1]):
                print('%6.1f %15.4f %15.2f' % (t, y, v))
    else:
        print('Error: one of (t_interr, y_interr) is None while the other not.')

# Set tspan to end at t_end1
tspan = np.arange(0, t_end1 + 1, 1.0, np.float)
#
# 1. Solve the problem without interrupt function i.e. compute until
#    the first root is found
#
solver = cvode.CVODE(rhs_fn, nr_rootfns=1, rootfn=root_fn)

(flag, t_retn, y_retn, t_interr, y_interr) = solver.solve(tspan, y0)
print_results(1, t_retn, y_retn, t_interr, y_interr) # flag is 2 == Root found

#
# 2. Solve the problem with interrupt function i.e. compute until
#    the final time is reached
#
solver = cvode.CVODE(rhs_fn, nr_rootfns=1, rootfn=root_fn,
                     interruptfn=interr_fn)

(flag, t_retn, y_retn, t_interr, y_interr) = solver.solve(tspan, y0)

if flag == 0: # not CV_SUCCESS_RETURN
    print_results(2, t_retn, y_retn, t_interr, y_interr)
else:
    print('Computation failed.')


# Set tspan to end at t_end2
tspan = np.arange(0, t_end2 + 1, 1.0, np.float)
#
# 3. Solve the problem without interrupt function i.e. compute until
#    the first root is found
#
solver = cvode.CVODE(rhs_fn, nr_rootfns=1, rootfn=root_fn)

(flag, t_retn, y_retn, t_interr, y_interr) = solver.solve(tspan, y0)
print_results(3, t_retn, y_retn, t_interr, y_interr)

#
# 4. Solve the problem with interrupt function i.e. compute until
#    the final time is reached
#
solver = cvode.CVODE(rhs_fn, nr_rootfns=1, rootfn=root_fn,
                     interruptfn=interr_fn)

(flag, t_retn, y_retn, t_interr, y_interr) = solver.solve(tspan, y0)
print_results(4, t_retn, y_retn, t_interr, y_interr)

#
# 5. Solve the problem without interrupt function i.e. compute until
#    the first root is found
#
solver = cvode.CVODE(rhs_fn, nr_rootfns=1, rootfn=root_fn,
                     interruptfn=interr_fn2)

(flag, t_retn, y_retn, t_interr, y_interr) = solver.solve(tspan, y0)
print_results(5, t_retn, y_retn, t_interr, y_interr)

#
# 6. Solve the problem with interrupt function i.e. compute until
#    the final time is reached
#
solver = cvode.CVODE(rhs_fn, nr_rootfns=1, rootfn=root_fn,
                     interruptfn=interr_fn3)

(flag, t_retn, y_retn, t_interr, y_interr) = solver.solve(tspan, y0)
# flag is 2 - root found
print_results(6, t_retn, y_retn, t_interr, y_interr)
