from __future__ import print_function
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
        diff = np.abs(self.solution.values.y[:, 0] - true_ys)
        rel_diff = diff / true_ys
        print("True solution:\n", true_ys)
        print("Integrated:\n", self.solution.values.y[:, 0])
        print("Difference:\n", diff)
        print("Relative:\n", rel_diff)
        assert rel_diff.max() < 1e-6

    def test_get_info_is_exposed_on_ode(self):
        self.ode.get_info()

    def test_get_info_returns_dict(self):
        assert isinstance(self.ode.get_info(), dict)

    def test_ode_exposes_num_rhs_evals(self):
        info = self.ode.get_info()
        print("ode.get_info() =\n", info)
        assert 'NumRhsEvals' in info
        assert info['NumRhsEvals'] > 0

class GetInfoTestSpils(unittest.TestCase):
    def setUp(self):
        self.ode = ode('cvode', rhs, linsolver="spgmr", old_api=False)
        self.solution = self.ode.solve(xs, np.array([1]))

    def test_ode_exposes_num_njtimes_evals(self):
        info = self.ode.get_info()
        print("ode.get_info() =\n", info)
        assert 'NumJtimesEvals' in info
        assert info['NumJtimesEvals'] > 0
