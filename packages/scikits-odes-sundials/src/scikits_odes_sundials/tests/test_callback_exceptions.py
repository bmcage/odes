"""
An exception raised in a user callback must come out of `solve` and `step`
unchanged, not as a `SystemError` and not swallowed.
"""
import pytest

from scikits_odes_sundials.cvode import CVODE
from scikits_odes_sundials.cvodes import CVODES
from scikits_odes_sundials.ida import IDA
from scikits_odes_sundials.idas import IDAS


class UserError(Exception):
    pass


EXCEPTIONS = [UserError, KeyboardInterrupt, SystemExit]


class RaiseAfter:
    """Run `fn` for the first `ok_calls` calls, then raise `exc`."""

    def __init__(self, fn, exc, ok_calls=0):
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


def cv_rhs(t, y, ydot):
    ydot[0] = -y[0]


def cv_jac(t, y, fy, J):
    J[0, 0] = -1


def cv_root(t, y, g):
    g[0] = 1


def ida_res(t, y, yp, r):
    r[0] = yp[0] + y[0]


def ida_jac(t, y, yp, r, cj, J):
    J[0, 0] = cj + 1


def ida_root(t, y, yp, g):
    g[0] = 1


def cv_problem(callback, exc):
    # CVODE.init_step calls rhs and jac once from Python as a check.
    cb = RaiseAfter(
        {"rhs": cv_rhs, "jac": cv_jac, "root": cv_root}[callback],
        exc,
        ok_calls=0 if callback == "root" else 1,
    )
    opts = {
        "rhs": dict(),
        "jac": dict(jacfn=cb, linsolver="dense"),
        "root": dict(rootfn=cb, nr_rootfns=1),
    }[callback]
    rhs = cb if callback == "rhs" else cv_rhs
    return cb, rhs, opts


def ida_problem(callback, exc):
    cb = RaiseAfter({"res": ida_res, "jac": ida_jac, "root": ida_root}[callback], exc)
    opts = {
        "res": dict(),
        "jac": dict(jacfn=cb),
        "root": dict(rootfn=cb, nr_rootfns=1),
    }[callback]
    res = cb if callback == "res" else ida_res
    return cb, res, opts


def assert_raises_original(cb, run):
    with pytest.raises(cb.exc) as excinfo:
        run()
    assert excinfo.value is cb.raised


@pytest.mark.parametrize("exc", EXCEPTIONS)
@pytest.mark.parametrize("err_handler", [None, lambda *args: None])
@pytest.mark.parametrize("callback", ["rhs", "jac", "root"])
@pytest.mark.parametrize("solver_cls", [CVODE, CVODES])
class TestCVODE:
    def solver(self, solver_cls, callback, exc, err_handler):
        cb, rhs, opts = cv_problem(callback, exc)
        if err_handler is not None:
            opts["err_handler"] = err_handler
        return cb, solver_cls(rhs, old_api=False, **opts)

    def test_solve(self, solver_cls, callback, exc, err_handler):
        cb, solver = self.solver(solver_cls, callback, exc, err_handler)
        assert_raises_original(cb, lambda: solver.solve([0, 1], [1.0]))

    def test_step(self, solver_cls, callback, exc, err_handler):
        cb, solver = self.solver(solver_cls, callback, exc, err_handler)
        solver.init_step(0, [1.0])
        assert_raises_original(cb, lambda: solver.step(1))


@pytest.mark.parametrize("exc", EXCEPTIONS)
@pytest.mark.parametrize("err_handler", [None, lambda *args: None])
@pytest.mark.parametrize("callback", ["res", "jac", "root"])
@pytest.mark.parametrize("solver_cls", [IDA, IDAS])
class TestIDA:
    def solver(self, solver_cls, callback, exc, err_handler, **extra):
        cb, res, opts = ida_problem(callback, exc)
        if err_handler is not None:
            opts["err_handler"] = err_handler
        return cb, solver_cls(res, old_api=False, **opts, **extra)

    def test_solve(self, solver_cls, callback, exc, err_handler):
        cb, solver = self.solver(solver_cls, callback, exc, err_handler)
        assert_raises_original(cb, lambda: solver.solve([0, 1], [1.0], [-1.0]))

    def test_step(self, solver_cls, callback, exc, err_handler):
        cb, solver = self.solver(solver_cls, callback, exc, err_handler)
        solver.init_step(0, [1.0], [-1.0])
        assert_raises_original(cb, lambda: solver.step(1))

    def test_calc_ic(self, solver_cls, callback, exc, err_handler):
        if callback == "root":
            pytest.skip("IDACalcIC does not call the root function")
        cb, solver = self.solver(
            solver_cls, callback, exc, err_handler, compute_initcond="yp0"
        )
        assert_raises_original(cb, lambda: solver.init_step(0, [1.0], [0.0]))
