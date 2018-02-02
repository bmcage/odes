# Authors: B. Malengier, russel (scipy trac)
from __future__ import print_function

"""
Tests for differential algebraic equation solvers.
Here we test onroot and ontstop
"""
import numpy as np

from numpy import (arange, zeros, array, dot, sqrt, cos, sin, allclose,
                    empty, alen)

from numpy.testing import TestCase, run_module_suite

from scikits.odes import dae
from scikits.odes.sundials.ida import StatusEnumIDA
from scikits.odes.sundials.common_defs import DTYPE

#data
g  = 9.81    # gravitational constant

Y0 = 1000.0  # Initial height
Y1 = 10.0    # Bottom height - when reached, changes to Y0 (teleport)
T1 = 10.0
v0 = 0.0     # Initial speed

#initial data at t=0, y[0] = Y, y[1] = \dot{Y}
y0 = [Y0, v0]
yp0 = [v0, -g]

t_end1 =  10.0 # Time of free fall for experiments 1,2
t_end2 = 100.0 # Time of free fall for experiments 3,4

atol = 1e-4
rtol = 1e-4

def rhs_fn(t, y, ydot, res):
    """ rhs equations for the problem """
    res[0] = ydot[0] - y[1]
    res[1] = ydot[1] + g

def root_fn(t, y, ydot, out):
    """ root function to check the object reached height Y1 """
    out[0] = Y1 - y[0]
    return 0

def root_fn2(t, y, ydot, out):
    """ root function to check the object reached height Y1 """
    out[0] = Y1 - y[0]
    out[1] = (t-10)*(t-20)*(t-30)
    return 0

def root_fn3(t, y, ydot, out):
    """ root function to check the object reached time 10 """
    out[0] = (t - 10)*(t-20)*(t-30)
    return 0

def onroot_va(t, y, ydot, solver):
    """
    onroot function to reset the solver back at the start, but keep the current
    velocity
    """
    # Teleport the object back to height Y0, but retain its speed
    solver.reinit_IC(t, [Y0, y[1]], ydot)

    return 0

def onroot_vb(t, y, ydot, solver):
    """
    onroot function to stop solver when root is found
    """
    return 1

def onroot_vc(t, y, ydot, solver):
    """
    onroot function to reset the solver back at the start, but keep the current
    velocity as long as the time is less than a given amount
    """
    if t > 28: # we have found 4 interruption points, so we stop
        return 1
    solver.reinit_IC(t, [Y0, y[1]], ydot)
    return 0

def onroot_vd(t, y, ydot, solver):
    """
    onroot function to just continue if time <28
    """
    if t > 28:
        return 1

    return 0

n=0

def ontstop_va(t, y, ydot, solver):
    """
    ontstop function to reset the solver back at the start, but keep the current
    velocity
    """
    # Teleport the object back to height Y0, but retain its speed
    global n
    solver.reinit_IC(t, [Y0, y[1]], ydot)
    n += 1
    solver.set_options(tstop=T1+n*10)

    return 0

def ontstop_vb(t, y, ydot, solver):
    """
    ontstop function to stop solver when tstop is reached
    """
    return 1

def ontstop_vc(t, y, ydot, solver):
    """
    ontstop function to reset the solver back at the start, but keep the current
    velocity as long as the time is less than a given amount
    """
    global n
    if t > 28: # we have found 3 interruption points, so we stop
        return 1
    solver.reinit_IC(t, [Y0, y[1]], ydot)
    n += 1
    solver.set_options(tstop=T1+n*10)

    return 0

class TestOn(TestCase):
    """
    Check integrate.dae
    """

    def test_ida_rootfn_noroot(self):
        #test calling sequence. End is reached before root is found
        tspan = np.arange(0, t_end1 + 1, 1.0, DTYPE)
        solver = dae('ida', rhs_fn, nr_rootfns=1, rootfn=root_fn,
                     old_api=False)
        soln = solver.solve(tspan, y0, yp0)
        assert soln.flag==StatusEnumIDA.SUCCESS, "ERROR: Error occurred"
        assert allclose([soln.values.t[-1], soln.values.y[-1,0], soln.values.y[-1,1]],
                        [10.0, 509.4995, -98.10],
                        atol=atol, rtol=rtol)

    def test_ida_rootfn(self):
        #test root finding and stopping: End is reached at a root
        tspan = np.arange(0, t_end2 + 1, 1.0, DTYPE)
        solver = dae('ida', rhs_fn, nr_rootfns=1, rootfn=root_fn,
                     old_api=False)
        soln = solver.solve(tspan, y0, yp0)
        assert soln.flag==StatusEnumIDA.ROOT_RETURN, "ERROR: Root not found!"
        assert allclose([soln.roots.t[0], soln.roots.y[0,0], soln.roots.y[0,1]],
                        [14.206856, 10.0000, -139.3693],
                        atol=atol, rtol=rtol)

    def test_ida_rootfnacc(self):
        #test root finding and accumilating: End is reached normally, roots stored
        tspan = np.arange(0, t_end2 + 1, 1.0, DTYPE)
        solver = dae('ida', rhs_fn, nr_rootfns=1, rootfn=root_fn,
                     onroot=onroot_va,
                     old_api=False)
        soln = solver.solve(tspan, y0, yp0)
        assert soln.flag==StatusEnumIDA.SUCCESS, "ERROR: Error occurred"
        assert allclose([soln.values.t[-1], soln.values.y[-1,0], soln.values.y[-1,1]],
                        [100.0, 459.999, -981.0000],
                        atol=atol, rtol=rtol)
        assert len(soln.roots.t) == 49, "ERROR: Did not find all 49 roots"
        assert allclose([soln.roots.t[-1], soln.roots.y[-1,0], soln.roots.y[-1,1]],
                        [99.447910, 10.0000, -975.5840],
                        atol=atol, rtol=rtol)

    def test_ida_rootfn_stop(self):
        #test root finding and stopping: End is reached at a root with a function
        tspan = np.arange(0, t_end2 + 1, 1.0, DTYPE)
        solver = dae('ida', rhs_fn, nr_rootfns=1, rootfn=root_fn,
                     onroot=onroot_vb,
                     old_api=False)
        soln = solver.solve(tspan, y0, yp0)
        assert soln.flag==StatusEnumIDA.ROOT_RETURN, "ERROR: Root not found!"
        assert allclose([soln.roots.t[-1], soln.roots.y[-1,0], soln.roots.y[-1,1]],
                        [14.206856, 10.0000, -139.3693],
                        atol=atol, rtol=rtol)

    def test_ida_rootfn_test(self):
        #test root finding and accumilating: End is reached after a number of root
        tspan = np.arange(0, t_end2 + 1, 1.0, DTYPE)
        solver = dae('ida', rhs_fn, nr_rootfns=1, rootfn=root_fn,
                     onroot=onroot_vc,
                     old_api=False)
        soln = solver.solve(tspan, y0, yp0)
        assert soln.flag==StatusEnumIDA.ROOT_RETURN, "ERROR: Not sufficient root found"
        assert allclose([soln.values.t[-1], soln.values.y[-1,0], soln.values.y[-1,1]],
                        [28.0, 124.4724, -274.6800],
                        atol=atol, rtol=rtol)
        assert len(soln.roots.t) == 4, "ERROR: Did not find all 4 roots"
        assert allclose([soln.roots.t[-1], soln.roots.y[-1,0], soln.roots.y[-1,1]],
                        [28.413692, 10.0000, -278.7383],
                        atol=atol, rtol=rtol)

    def test_ida_rootfn_two(self):
        #test two root finding
        tspan = np.arange(0, t_end2 + 1, 1.0, DTYPE)
        solver = dae('ida', rhs_fn, nr_rootfns=2, rootfn=root_fn2,
                     onroot=onroot_vc,
                     old_api=False)
        soln = solver.solve(tspan, y0, yp0)
        assert soln.flag==StatusEnumIDA.ROOT_RETURN, "ERROR: Not sufficient root found"
        assert allclose([soln.values.t[-1], soln.values.y[-1,0], soln.values.y[-1,1]],
                        [28.0, 106.4753, -274.6800],
                        atol=atol, rtol=rtol)
        assert len(soln.roots.t) == 5, "ERROR: Did not find all 4 roots"
        assert allclose([soln.roots.t[-1], soln.roots.y[-1,0], soln.roots.y[-1,1]],
                        [28.349052, 10.0000, -278.1042],
                        atol=atol, rtol=rtol)

    def test_ida_rootfn_end(self):
        #test root finding with root at endtime
        tspan = np.arange(0, 30 + 1, 1.0, DTYPE)
        solver = dae('ida', rhs_fn, nr_rootfns=1, rootfn=root_fn3,
                     onroot=onroot_vc,
                     old_api=False)
        soln = solver.solve(tspan, y0, yp0)
        assert soln.flag==StatusEnumIDA.ROOT_RETURN, "ERROR: Not sufficient root found"
        assert allclose([soln.values.t[-1], soln.values.y[-1,0], soln.values.y[-1,1]],
                        [30.0, -1452.5024, -294.3000],
                        atol=atol, rtol=rtol)
        assert len(soln.roots.t) == 3, "ERROR: Did not find all 4 roots"
        assert allclose([soln.roots.t[-1], soln.roots.y[-1,0], soln.roots.y[-1,1]],
                        [30.0, -1452.5024, -294.3000],
                        atol=atol, rtol=rtol)

    def test_ida_tstopfn_notstop(self):
        #test calling sequence. End is reached before tstop is found
        global n
        n = 0
        tspan = np.arange(0, t_end1 + 1, 1.0, DTYPE)
        solver = dae('ida', rhs_fn, tstop=T1+1, ontstop=ontstop_va,
                     old_api=False)
        soln = solver.solve(tspan, y0, yp0)
        assert soln.flag==StatusEnumIDA.SUCCESS, "ERROR: Error occurred"
        assert allclose([soln.values.t[-1], soln.values.y[-1,0], soln.values.y[-1,1]],
                        [10.0, 509.4995, -98.10],
                        atol=atol, rtol=rtol)

    def test_ida_tstopfn(self):
        #test tstop finding and stopping: End is reached at a tstop
        global n
        n = 0
        tspan = np.arange(0, t_end2 + 1, 1.0, DTYPE)
        solver = dae('ida', rhs_fn, tstop=T1,
                     old_api=False)
        soln = solver.solve(tspan, y0, yp0)
        assert soln.flag==StatusEnumIDA.TSTOP_RETURN, "ERROR: Tstop not found!"
        assert allclose([soln.tstop.t[0], soln.tstop.y[0,0], soln.tstop.y[0,1]],
                        [10.0, 509.4995, -98.10],
                        atol=atol, rtol=rtol)
        assert allclose([soln.values.t[-1], soln.values.y[-1,0], soln.values.y[-1,1]],
                        [10.0, 509.4995, -98.10],
                        atol=atol, rtol=rtol)

    def test_ida_tstopfnacc(self):
        #test tstop finding and accumilating: End is reached normally, tstop stored
        global n
        n = 0
        tspan = np.arange(0, t_end2 + 1, 1.0, DTYPE)
        solver = dae('ida', rhs_fn, tstop=T1, ontstop=ontstop_va,
                     old_api=False)
        soln = solver.solve(tspan, y0, yp0)
        assert soln.flag==StatusEnumIDA.SUCCESS, "ERROR: Error occurred"
        assert allclose([soln.values.t[-1], soln.values.y[-1,0], soln.values.y[-1,1]],
                        [100.0, -8319.5023, -981.00],
                        atol=atol, rtol=rtol)
        assert len(soln.tstop.t) == 9, "ERROR: Did not find all tstop"
        assert allclose([soln.tstop.t[-1], soln.tstop.y[-1,0], soln.tstop.y[-1,1]],
                        [90.0, -7338.5023, -882.90],
                        atol=atol, rtol=rtol)

    def test_ida_tstopfn_stop(self):
        #test calling sequence. End is reached at a tstop
        global n
        n = 0
        tspan = np.arange(0, t_end2 + 1, 1.0, DTYPE)
        solver = dae('ida', rhs_fn, tstop=T1, ontstop=ontstop_vb,
                     old_api=False)

        soln = solver.solve(tspan, y0, yp0)
        assert soln.flag==StatusEnumIDA.TSTOP_RETURN, "ERROR: Error occurred"
        assert allclose([soln.values.t[-1], soln.values.y[-1,0], soln.values.y[-1,1]],
                        [10.0, 509.4995, -98.10],
                        atol=atol, rtol=rtol)
        assert len(soln.tstop.t) == 1, "ERROR: Did not find all tstop"

        assert len(soln.values.t) == 11, "ERROR: Did not find all output"
        assert allclose([soln.tstop.t[-1], soln.tstop.y[-1,0], soln.tstop.y[-1,1]],
                        [10.0, 509.4995, -98.10],
                        atol=atol, rtol=rtol)

    def test_ida_tstopfn_test(self):
        #test calling sequence. tsop function continues up to a time
        global n
        n = 0
        tspan = np.arange(0, t_end2 + 1, 1.0, DTYPE)
        solver = dae('ida', rhs_fn, tstop=T1, ontstop=ontstop_vc,
                     old_api=False)

        soln = solver.solve(tspan, y0, yp0)
        assert soln.flag==StatusEnumIDA.TSTOP_RETURN, "ERROR: Error occurred"
        assert allclose([soln.values.t[-1], soln.values.y[-1,0], soln.values.y[-1,1]],
                        [30.0, -1452.5024, -294.30],
                        atol=atol, rtol=rtol)
        assert len(soln.tstop.t) == 3, "ERROR: Did not find all tstop"
        assert len(soln.values.t) == 31, "ERROR: Did not find all output"
        assert allclose([soln.tstop.t[-1], soln.tstop.y[-1,0], soln.tstop.y[-1,1]],
                        [30.0, -1452.5024, -294.30],
                        atol=atol, rtol=rtol)
if __name__ == "__main__":
    try:
        run_module_suite()
    except NameError:
        test = TestOn()
        test.test_ida_rootfn_noroot()
        test.test_ida_rootfn()
        test.test_ida_rootfnacc()
        test.test_ida_rootfn_stop()
        test.test_ida_rootfn_test()
