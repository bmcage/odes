# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 12:58:13 2016

@author: benny
"""

from __future__ import division, print_function, absolute_import

import numpy as np

from numpy.testing import (
    assert_, TestCase, run_module_suite, assert_array_almost_equal,
    assert_raises, assert_allclose, assert_array_equal, assert_equal)
from scikits.odes import ode
from scikits.odes.dopri5 import StatusEnumDOP


class SimpleOscillator():
    r"""
    Free vibration of a simple oscillator::
        m \ddot{u} + k u = 0, u(0) = u_0 \dot{u}(0) \dot{u}_0
    Solution::
        u(t) = u_0*cos(sqrt(k/m)*t)+\dot{u}_0*sin(sqrt(k/m)*t)/sqrt(k/m)
    """
    stop_t = 1 + 0.09
    y0 = np.array([1.0, 0.1], float)

    k = 4.0
    m = 1.0
    atol = 1e-6
    rtol = 1e-6

    def f(self, t, y, out):
        tmp = np.zeros((2, 2), float)
        tmp[0, 1] = 1.0
        tmp[1, 0] = -self.k / self.m
        out[:] = np.dot(tmp, y)[:]

    def verify(self, t, y):
        omega = np.sqrt(self.k / self.m)
        u = self.y0[0]*np.cos(omega*t) + self.y0[1]*np.sin(omega*t)/omega
        return np.allclose(u, y[0], atol=self.atol, rtol=self.rtol)

class TestDop(TestCase):

    def test_dop_oscil(self):
        #test calling sequence. End is reached before root is found
        prob = SimpleOscillator()
        tspan = [0, prob.stop_t]
        solver = ode('dopri5', prob.f)
        soln = solver.solve(tspan, prob.y0)
        assert soln.flag==StatusEnumDOP.SUCCESS, "ERROR: Error occurred"
        assert prob.verify(prob.stop_t, soln.values.y[1])

        solver = ode('dop853', prob.f)
        soln = solver.solve(tspan, prob.y0)
        assert soln.flag==StatusEnumDOP.SUCCESS, "ERROR: Error occurred"
        assert prob.verify(prob.stop_t, soln.values.y[1])
