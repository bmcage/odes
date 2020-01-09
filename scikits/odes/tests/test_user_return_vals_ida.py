
from numpy.testing import TestCase, run_module_suite

from .. import dae
from ..sundials.ida import StatusEnumIDA

def normal_rhs(t, y, ydot, res):
    res[0] = ydot - t

def complex_rhs(t, y, ydot, res):
    res[0] = ydot - t + y

def rhs_with_return(t, y, ydot, res):
    res[0] = ydot - t
    return 0

def rhs_problem_late(t, y, ydot, res):
    res[0] = ydot - t
    if t > 0.5:
        return 1

def rhs_problem_immediate(t, y, ydot, res):
    return 1

def rhs_error_late(t, y, ydot, res):
    res[0] = ydot - t
    if t > 0.5:
        return -1

def rhs_error_immediate(t, y, ydot, res):
    return -1

def normal_root(t, y, y_dot, g):
    g[0] = 1

def root_with_return(t, y, y_dot, g):
    g[0] = 1
    return 0

def root_late(t, y, y_dot, g):
    g[0] = 1
    if t > 0.5:
        g[0] = 0

def root_immediate(t, y, y_dot, g):
    g[0] = 0

def root_error_late(t, y, y_dot, g):
    g[0] = t
    if t > 0.5:
        return 1

def root_error_immediate(t, y, y_dot, g):
    return 1

def normal_jac(t, y, ydot, residual, cj, J):
    J[0][0] = cj

def jac_with_return(t, y, ydot, residual, cj, J):
    J[0][0] = cj
    return 0

def jac_problem_late(t, y, ydot, residual, cj, J):
    J[0][0] = cj + 1
    if t > 0:
        return 1

def jac_problem_immediate(t, y, ydot, residual, cj, J):
    return 1

def jac_error_late(t, y, ydot, residual, cj, J):
    J[0][0] = cj + 1
    if t > 0:
        return -1

def jac_error_immediate(t, y, ydot, residual, cj, J):
    return -1

class TestIdaReturn(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestIdaReturn, self).__init__(*args, **kwargs)
        self.solvername = "ida"

    def test_normal_rhs(self):
        solver = dae(self.solvername, normal_rhs, old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.SUCCESS,
            soln.flag
        )

    def test_rhs_with_return(self):
        solver = dae(self.solvername, rhs_with_return, old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.SUCCESS,
            soln.flag
        )

    def test_rhs_problem_late(self):
        solver = dae(self.solvername, rhs_problem_late, old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.TOO_MUCH_WORK,
            soln.flag
        )

    def test_rhs_problem_immediate(self):
        solver = dae(self.solvername, rhs_problem_immediate, old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.REP_RES_ERR,
            soln.flag
        )

    def test_rhs_error_late(self):
        solver = dae(self.solvername, rhs_error_late, old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.RES_FAIL,
            soln.flag
        )

    def test_rhs_error_immediate(self):
        solver = dae(self.solvername, rhs_error_immediate, old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.RES_FAIL,
            soln.flag
        )

    def test_normal_root(self):
        solver = dae(self.solvername, normal_rhs, rootfn=normal_root, nr_rootfns=1,
                old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.SUCCESS,
            soln.flag
        )

    def test_root_with_return(self):
        solver = dae(self.solvername, normal_rhs, rootfn=root_with_return, nr_rootfns=1,
                old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.SUCCESS,
            soln.flag
        )

    def test_root_late(self):
        solver = dae(self.solvername, normal_rhs, rootfn=root_late, nr_rootfns=1,
                old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.ROOT_RETURN,
            soln.flag
        )

    def test_root_immediate(self):
        solver = dae(self.solvername, normal_rhs, rootfn=root_immediate, nr_rootfns=1,
                old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.SUCCESS,
            soln.flag
        )

    def test_root_error_late(self):
        solver = dae(self.solvername, normal_rhs, rootfn=root_error_late, nr_rootfns=1,
                old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.RTFUNC_FAIL,
            soln.flag
        )

    def test_root_error_immediate(self):
        solver = dae(self.solvername, normal_rhs, rootfn=root_error_immediate,
                nr_rootfns=1, old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.RTFUNC_FAIL,
            soln.flag
        )

    def test_normal_jac(self):
        solver = dae(self.solvername, normal_rhs, jacfn=normal_jac, old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.SUCCESS,
            soln.flag
        )

    def test_jac_with_return(self):
        solver = dae(self.solvername, normal_rhs, jacfn=jac_with_return, old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.SUCCESS,
            soln.flag
        )

    def test_jac_problem_late(self):
        solver = dae(self.solvername, complex_rhs, jacfn=jac_problem_late, old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.CONV_FAIL,
            soln.flag
        )

    def test_jac_problem_immediate(self):
        solver = dae(self.solvername, normal_rhs, jacfn=jac_problem_immediate,
                old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.CONV_FAIL,
            soln.flag
        )

    def test_jac_error_late(self):
        solver = dae(self.solvername, complex_rhs, jacfn=jac_error_late, old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.LSETUP_FAIL,
            soln.flag
        )

    def test_jac_error_immediate(self):
        solver = dae(self.solvername, normal_rhs, jacfn=jac_error_immediate, old_api=False)
        soln = solver.solve([0, 1], [1], [0])
        self.assertEqual(
            StatusEnumIDA.LSETUP_FAIL,
            soln.flag
        )


class TestIdasReturn(TestIdaReturn):
    def __init__(self, *args, **kwargs):
        super(TestIdasReturn, self).__init__(*args, **kwargs)
        self.solvername = "idas"

if __name__ == "__main__":
    try:
        run_module_suite()
    except NameError:
        test = TestIdaReturn()
        test.test_normal_rhs()
