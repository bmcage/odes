import numpy as np
import unittest
from scikits.odes import ode

xs = np.linspace(1, 10, 10)


def true_y(x):
    """ just a dummy this test module is not about the integration itself """
    return np.power(x, 2)
    #return np.sin(x) / (x + 0.1)


def rhs(x, y, ydot):
    ydot[:] = 2 * x
    #ydot[:] = (np.cos(x) * (x + 0.1) - np.sin(x)) / np.pow((x + 0.1), 2)


class GetInfoTest(unittest.TestCase):
    def setUp(self):
        self.ode = ode('cvode', rhs, old_api=False)
        self.solution = self.ode.solve(xs, np.array([1]))

    def test_we_integrated_correctly(self):
        true_ys = true_y(xs)
        rel_diff = np.abs(self.solution.values.y - true_ys) / true_ys
        print "Relative difference is d_max={}.".format(rel_diff.max())
        assert rel_diff.max() < 1e-6

    def test_get_info_is_exposed_on_ode(self):
        ode.get_info()
