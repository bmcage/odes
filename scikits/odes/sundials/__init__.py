#
# odes - Extra ode integrators
#

import inspect

class CVODESolveException(Exception):
    """Base class for exceptions raised by ``CVODE.validate_flags``."""
    def __init__(self, soln):
        self.soln = soln
        self.args = (self._message.format(soln),)

class CVODESolveFailed(CVODESolveException):
    """``CVODE.solve`` failed to reach endpoint"""
    _message = (
        "Solver failed with flag {0.flag} and finished at {0.errors.t}"
        "with values {0.errors.y}."
    )

class CVODESolveFoundRoot(CVODESolveException):
    """``CVODE.solve`` found a root"""
    _message = "Solver found a root at {0.roots.t[0]}."

class CVODESolveReachedTSTOP(CVODESolveException):
    """``CVODE.solve`` reached the endpoint specified by tstop."""
    _message = "Solver reached tstop at {0.tstop.t[0]}."

class IDASolveException(Exception):
    """Base class for exceptions raised by ``IDA.validate_flags``."""
    def __init__(self, soln):
        self.soln = soln
        self.args = (self._message.format(soln),)

class IDASolveFailed(IDASolveException):
    """``IDA.solve`` failed to reach endpoint"""
    _message = (
        "Solver failed with flag {0.flag} and finished at {0.errors.t}"
        "with values {0.errors.y} and derivatives {0.errors.ydot}."
    )

class IDASolveFoundRoot(IDASolveException):
    """``IDA.solve`` found a root"""
    _message = "Solver found a root at {0.roots.t[0]}."

class IDASolveReachedTSTOP(IDASolveException):
    """``IDA.solve`` reached the endpoint specified by tstop."""
    _message = "Solver reached tstop at {0.tstop.t[0]}."


def _get_num_args(func):
    """
    Python 2/3 compatible method of getting number of args that `func` accepts
    """
    if hasattr(inspect, "signature"):
        sig = inspect.signature(func)
        numargs = 0
        for param in sig.parameters.values():
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                numargs += 1
        return numargs
    else:
        return len(inspect.getargspec(func).args)
