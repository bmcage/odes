
from numpy.testing import TestCase, run_module_suite

from .. import ode
from ..sundials.cvode import StatusEnum

def normal_rhs(t, y, ydot):
    ydot[0] = t

def rhs_with_return(t, y, ydot):
    ydot[0] = t
    return 0

def rhs_problem_late(t, y, ydot):
    ydot[0] = t
    if t > 0.5:
        return 1

def rhs_problem_immediate(t, y, ydot):
    return 1

def rhs_error_late(t, y, ydot):
    ydot[0] = t
    if t > 0.5:
        return -1

def rhs_error_immediate(t, y, ydot):
    return -1


def normal_root(t, y, g):
    g[0] = 1

def root_with_return(t, y, g):
    g[0] = 1
    return 0

def root_late(t, y, g):
    g[0] = 1
    if t > 0.5:
        g[0] = 0

def root_immediate(t, y, g):
    g[0] = 0

def root_error_late(t, y, g):
    g[0] = t
    if t > 0.5:
        return 1

def root_error_immediate(t, y, g):
    return 1

#jacfn
#prec
#jactimes

class TestCVodeReturn(TestCase):
    def test_normal_rhs(self):
        solver = ode("cvode", normal_rhs)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.SUCCESS,
            soln[0]
        )

    def test_rhs_with_return(self):
        solver = ode("cvode", rhs_with_return)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.SUCCESS,
            soln[0]
        )

    def test_rhs_problem_late(self):
        solver = ode("cvode", rhs_problem_late)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.TOO_MUCH_WORK,
            soln[0]
        )
    def test_rhs_problem_immediate(self):
        solver = ode("cvode", rhs_problem_immediate)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.FIRST_RHSFUNC_ERR,
            soln[0]
        )
    def test_rhs_error_late(self):
        solver = ode("cvode", rhs_error_late)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.RHSFUNC_FAIL,
            soln[0]
        )
    def test_rhs_error_immediate(self):
        solver = ode("cvode", rhs_error_immediate)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.RHSFUNC_FAIL,
            soln[0]
        )

    def test_normal_root(self):
        solver = ode("cvode", normal_rhs, rootfn=normal_root, nr_rootfns=1)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.SUCCESS,
            soln[0]
        )
    def test_root_with_return(self):
        solver = ode("cvode", normal_rhs, rootfn=root_with_return, nr_rootfns=1)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.SUCCESS,
            soln[0]
        )
    def test_root_late(self):
        solver = ode("cvode", normal_rhs, rootfn=root_late, nr_rootfns=1)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.ROOT_RETURN,
            soln[0]
        )
    def test_root_immediate(self):
        solver = ode("cvode", normal_rhs, rootfn=root_immediate, nr_rootfns=1)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.SUCCESS,
            soln[0]
        )
    def test_root_error_late(self):
        solver = ode("cvode", normal_rhs, rootfn=root_error_late, nr_rootfns=1)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.RTFUNC_FAIL,
            soln[0]
        )
    def test_root_error_immediate(self):
        solver = ode("cvode", normal_rhs, rootfn=root_error_immediate, nr_rootfns=1)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.RTFUNC_FAIL,
            soln[0]
        )


if __name__ == "__main__":
    try:
        run_module_suite()
    except NameError:
        test = TestOn()
