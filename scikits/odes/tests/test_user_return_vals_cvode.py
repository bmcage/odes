
from numpy.testing import TestCase, run_module_suite

from .. import ode
from ..sundials.cvode import StatusEnum

def normal_rhs(t, y, ydot):
    ydot[0] = t

def complex_rhs(t, y, ydot):
    ydot[0] = t - y

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

def normal_jac(t, y, fy, J):
    J[0][0] = 0

def jac_with_return(t, y, fy, J):
    J[0][0] = 0
    return 0

def jac_problem_late(t, y, fy, J):
    J[0][0] = 1
    if t > 0:
        return 1

def jac_problem_immediate(t, y, fy, J):
    return 1

def jac_error_late(t, y, fy, J):
    J[0][0] = 1
    if t > 0:
        return -1

def jac_error_immediate(t, y, fy, J):
    return -1


def normal_jac_vec(v, Jv, t, y):
    Jv[0] = 0

def jac_vec_with_return(v, Jv, t, y):
    Jv[0] = 0
    return None

def jac_vec_problem_late(v, Jv, t, y):
    Jv[0] = v[0]
    if t > 0:
        return 1

def jac_vec_problem_immediate(v, Jv, t, y):
    return 1

def jac_vec_error_late(v, Jv, t, y):
    Jv[0] = v[0]
    if t > 0:
        return -1

def jac_vec_error_immediate(v, Jv, t, y):
    return -1

class TestCVodeReturn(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestCVodeReturn, self).__init__(*args, **kwargs)
        self.solvername = "cvode"

    def test_normal_rhs(self):
        solver = ode(self.solvername, normal_rhs, old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.SUCCESS,
            soln.flag
        )

    def test_rhs_with_return(self):
        solver = ode(self.solvername, rhs_with_return, old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.SUCCESS,
            soln.flag
        )

    def test_rhs_problem_late(self):
        solver = ode(self.solvername, rhs_problem_late, old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.TOO_MUCH_WORK,
            soln.flag
        )

    def test_rhs_problem_immediate(self):
        solver = ode(self.solvername, rhs_problem_immediate, old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.FIRST_RHSFUNC_ERR,
            soln.flag
        )

    def test_rhs_error_late(self):
        solver = ode(self.solvername, rhs_error_late, old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.RHSFUNC_FAIL,
            soln.flag
        )

    def test_rhs_error_immediate(self):
        solver = ode(self.solvername, rhs_error_immediate, old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.RHSFUNC_FAIL,
            soln.flag
        )

    def test_normal_root(self):
        solver = ode(self.solvername, normal_rhs, rootfn=normal_root, nr_rootfns=1,
                old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.SUCCESS,
            soln.flag
        )

    def test_root_with_return(self):
        solver = ode(self.solvername, normal_rhs, rootfn=root_with_return, nr_rootfns=1,
                old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.SUCCESS,
            soln.flag
        )

    def test_root_late(self):
        solver = ode(self.solvername, normal_rhs, rootfn=root_late, nr_rootfns=1,
                old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.ROOT_RETURN,
            soln.flag
        )

    def test_root_immediate(self):
        solver = ode(self.solvername, normal_rhs, rootfn=root_immediate, nr_rootfns=1,
                old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.SUCCESS,
            soln.flag
        )

    def test_root_error_late(self):
        solver = ode(self.solvername, normal_rhs, rootfn=root_error_late, nr_rootfns=1,
                old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.RTFUNC_FAIL,
            soln.flag
        )

    def test_root_error_immediate(self):
        solver = ode(self.solvername, normal_rhs, rootfn=root_error_immediate,
                nr_rootfns=1, old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.RTFUNC_FAIL,
            soln.flag
        )

    def test_normal_jac(self):
        solver = ode(self.solvername, normal_rhs, jacfn=normal_jac, old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.SUCCESS,
            soln.flag
        )

    def test_jac_with_return(self):
        solver = ode(self.solvername, normal_rhs, jacfn=jac_with_return, old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.SUCCESS,
            soln.flag
        )

    def test_jac_problem_late(self):
        solver = ode(self.solvername, complex_rhs, jacfn=jac_problem_late, old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.CONV_FAILURE,
            soln.flag
        )

    def test_jac_problem_immediate(self):
        solver = ode(self.solvername, normal_rhs, jacfn=jac_problem_immediate,
                old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.CONV_FAILURE,
            soln.flag
        )

    def test_jac_error_late(self):
        solver = ode(self.solvername, complex_rhs, jacfn=jac_error_late, old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.LSETUP_FAIL,
            soln.flag
        )

    def test_jac_error_immediate(self):
        solver = ode(self.solvername, normal_rhs, jacfn=jac_error_immediate,
                old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.LSETUP_FAIL,
            soln.flag
        )


    def test_normal_jac_vec(self):
        solver = ode(self.solvername, normal_rhs, jac_times_vecfn=normal_jac_vec, old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.SUCCESS,
            soln.flag
        )

    def test_jac_vec_with_return(self):
        solver = ode(self.solvername, normal_rhs, jac_times_vecfn=jac_vec_with_return, linsolver="spgmr", old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.SUCCESS,
            soln.flag
        )

    def test_jac_vec_problem_late(self):
        solver = ode(self.solvername, complex_rhs, jac_times_vecfn=jac_vec_problem_late, linsolver="spgmr", old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.TOO_MUCH_WORK,
            soln.flag
        )

    def test_jac_vec_problem_immediate(self):
        solver = ode(self.solvername, normal_rhs,
                jac_times_vecfn=jac_vec_problem_immediate, linsolver="spgmr",
                old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.TOO_MUCH_WORK,
            soln.flag
        )

    def test_jac_vec_error_late(self):
        solver = ode(self.solvername, complex_rhs, jac_times_vecfn=jac_vec_error_late,
                linsolver="spgmr", old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.LSOLVE_FAIL,
            soln.flag
        )

    def test_jac_vec_error_immediate(self):
        solver = ode(self.solvername, normal_rhs, jac_times_vecfn=jac_vec_error_immediate, linsolver="spgmr",
                old_api=False)
        soln = solver.solve([0, 1], [1])
        self.assertEqual(
            StatusEnum.LSOLVE_FAIL,
            soln.flag
        )

class TestCVodesReturn(TestCVodeReturn):
    def __init__(self, *args, **kwargs):
        super(TestCVodesReturn, self).__init__(*args, **kwargs)
        self.solvername = "cvodes"

if __name__ == "__main__":
    try:
        run_module_suite()
    except NameError:
        test = TestCVodeReturn()
        TestCVodeReturn.test_normal_rhs()
