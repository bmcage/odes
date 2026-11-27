"""
An exception raised in a user callback must come out of `solve`, `step` and
`init_step` unchanged, not as a `SystemError` and not swallowed, and SUNDIALS
must not call the failing callback again.
"""
import pytest

from scikits_odes_sundials.cvode import CVODE
from scikits_odes_sundials.cvodes import CVODES
from scikits_odes_sundials.ida import IDA
from scikits_odes_sundials.idas import IDAS


class UserError(Exception):
    pass


class UserBaseError(BaseException):
    pass


EXCEPTIONS = [UserError, UserBaseError, SystemExit]
ERR_HANDLERS = [None, lambda *args: None]


class RaiseAfter:
    """Run `fn` for the first `ok_calls` calls, then raise `exc`."""

    def __init__(self, fn, exc, ok_calls):
        self.fn = fn
        self.exc = exc
        self.ok_calls = ok_calls
        self.calls = 0
        self.raised = None

    def __call__(self, *args):
        self.calls += 1
        if self.calls > self.ok_calls:
            self.raised = self.exc("raised in callback")
            raise self.raised
        return self.fn(*args)


def assert_raises_original(cb, run):
    with pytest.raises(cb.exc) as excinfo:
        run()
    assert excinfo.value is cb.raised
    assert cb.calls == cb.ok_calls + 1


# y' = -y
def cv_rhs(t, y, ydot):
    ydot[0] = -y[0]


def cv_jac(t, y, fy, J):
    J[0, 0] = -1


def cv_root(t, y, g):
    g[0] = 1


def cv_prec_setup(t, y, jok, jcur, gamma):
    jcur.value = True


def cv_prec_solve(t, y, r, z, gamma, delta, lr):
    z[0] = r[0] / (1 + gamma)


def cv_jac_times_setup(t, y, fy):
    pass


def cv_jac_times_vec(v, Jv, t, y):
    Jv[0] = -v[0]


CV_PREC = {"linsolver": "spgmr", "precond_type": "left",
           "prec_setupfn": cv_prec_setup, "prec_solvefn": cv_prec_solve}
CV_JTIMES = {"linsolver": "spgmr", "jac_times_setupfn": cv_jac_times_setup,
             "jac_times_vecfn": cv_jac_times_vec}

# name: (function, option, other options). CVODE.init_step calls rhs and
# jac once from Python as a check.
CV_CASES = {
    "rhs": (cv_rhs, None, {}),
    "jac": (cv_jac, "jacfn", {"linsolver": "dense"}),
    "root": (cv_root, "rootfn", {"nr_rootfns": 1}),
    "prec_setup": (cv_prec_setup, "prec_setupfn", CV_PREC),
    "prec_solve": (cv_prec_solve, "prec_solvefn", CV_PREC),
    "jac_times_setup": (cv_jac_times_setup, "jac_times_setupfn", CV_JTIMES),
    "jac_times_vec": (cv_jac_times_vec, "jac_times_vecfn", CV_JTIMES),
}


# y' + y = 0
def ida_res(t, y, yp, r):
    r[0] = yp[0] + y[0]


def ida_jac(t, y, yp, r, cj, J):
    J[0, 0] = cj + 1


def ida_root(t, y, yp, g):
    g[0] = 1


def ida_jac_times_setup(t, y, yp, r, cj):
    pass


def ida_jac_times_vec(t, y, yp, r, v, Jv, cj):
    Jv[0] = (cj + 1) * v[0]


IDA_JTIMES = {"linsolver": "spgmr", "jac_times_setupfn": ida_jac_times_setup,
              "jac_times_vecfn": ida_jac_times_vec}

IDA_CASES = {
    "res": (ida_res, None, {}),
    "jac": (ida_jac, "jacfn", {}),
    "root": (ida_root, "rootfn", {"nr_rootfns": 1}),
    "jac_times_setup": (ida_jac_times_setup, "jac_times_setupfn", IDA_JTIMES),
    "jac_times_vec": (ida_jac_times_vec, "jac_times_vecfn", IDA_JTIMES),
}


def make_solver(solver_cls, case, exc, err_handler, ok_calls=0, **extra):
    fn, option, opts = case
    cb = RaiseAfter(fn, exc, ok_calls)
    opts = dict(opts, old_api=False, **extra)
    if option is None:
        func = cb
    else:
        func = cv_rhs if issubclass(solver_cls, CVODE) else ida_res
        opts[option] = cb
    if err_handler is not None:
        opts["err_handler"] = err_handler
    return cb, solver_cls(func, **opts)


@pytest.mark.parametrize("exc", EXCEPTIONS)
@pytest.mark.parametrize("err_handler", ERR_HANDLERS)
@pytest.mark.parametrize("name", CV_CASES)
@pytest.mark.parametrize("solver_cls", [CVODE, CVODES])
class TestCVODE:
    def solver(self, solver_cls, name, exc, err_handler):
        ok_calls = 1 if name in ("rhs", "jac") else 0
        return make_solver(solver_cls, CV_CASES[name], exc, err_handler, ok_calls)

    def test_solve(self, solver_cls, name, exc, err_handler):
        cb, solver = self.solver(solver_cls, name, exc, err_handler)
        assert_raises_original(cb, lambda: solver.solve([0, 1], [1.0]))

    def test_step(self, solver_cls, name, exc, err_handler):
        cb, solver = self.solver(solver_cls, name, exc, err_handler)
        solver.init_step(0, [1.0])
        assert_raises_original(cb, lambda: solver.step(1))


@pytest.mark.parametrize("exc", EXCEPTIONS)
@pytest.mark.parametrize("err_handler", ERR_HANDLERS)
@pytest.mark.parametrize("name", IDA_CASES)
@pytest.mark.parametrize("solver_cls", [IDA, IDAS])
class TestIDA:
    def test_solve(self, solver_cls, name, exc, err_handler):
        cb, solver = make_solver(solver_cls, IDA_CASES[name], exc, err_handler)
        assert_raises_original(cb, lambda: solver.solve([0, 1], [1.0], [-1.0]))

    def test_step(self, solver_cls, name, exc, err_handler):
        cb, solver = make_solver(solver_cls, IDA_CASES[name], exc, err_handler)
        solver.init_step(0, [1.0], [-1.0])
        assert_raises_original(cb, lambda: solver.step(1))

    def test_calc_ic(self, solver_cls, name, exc, err_handler):
        if name == "root":
            pytest.skip("IDACalcIC does not call the root function")
        cb, solver = make_solver(solver_cls, IDA_CASES[name], exc, err_handler,
                                 compute_initcond="yp0")
        assert_raises_original(cb, lambda: solver.init_step(0, [1.0], [0.0]))
