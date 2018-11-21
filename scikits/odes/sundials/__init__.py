#
# odes - Extra ode integrators
#

import inspect
from logging import getLogger

logger = getLogger(__name__)
DEFAULT_LOG_FORMAT = "SUNDIALS message in %s:%s: %s"


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


def drop_all_error_handler(error_code, module, func, msg, user_data):
    """
    Drop all CVODE/IDA messages, rather than printing them.

    Examples
    --------
    >>> scikits.odes.ode('cvode', rhsfuc, err_handler=drop_all_error_handler)
    """
    # pylint: disable=unused-argument
    pass


def log_error_handler(error_code, module, func, msg, user_data):
    """
    Log all CVODE/IDA messages using the builtin python logging.

    Examples
    --------
    >>> scikits.odes.ode('cvode', rhsfuc, err_handler=log_error_handler)
    """
    # pylint: disable=unused-argument
    if error_code > 0:
        logger.warning(DEFAULT_LOG_FORMAT, module, func, msg)
    else:
        logger.error(DEFAULT_LOG_FORMAT, module, func, msg)


def onroot_continue(*args):
    """
    Always continue after finding root.

    Examples
    --------
    >>> scikits.odes.ode(
    ...     'cvode', rhsfuc, rootfn=rootfn, nr_rootfns=nroots,
    ...     onroot=onroot_continue
    ... )
    """
    # pylint: disable=unused-argument
    return 0


def onroot_stop(*args):
    """
    Always stop after finding root.

    Examples
    --------
    >>> scikits.odes.ode(
    ...     'cvode', rhsfuc, rootfn=rootfn, nr_rootfns=nroots,
    ...     onroot=onroot_stop
    ... )
    """
    # pylint: disable=unused-argument
    return 1


def ontstop_continue(*args):
    """
    Always continue after finding tstop.

    Examples
    --------
    >>> scikits.odes.ode(
    ...     'cvode', rhsfuc, tstop=tstop, ontstop=ontstop_continue
    ... )
    """
    # pylint: disable=unused-argument
    return 0


def ontstop_stop(*args):
    """
    Always stop after finding tstop.

    Examples
    --------
    >>> scikits.odes.ode('cvode', rhsfuc, tstop=tstop, ontstop=ontstop_stop)
    """
    # pylint: disable=unused-argument
    return 1
