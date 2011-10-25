#
# odes - Extra ode integrators
#

from .info import __doc__

from .dae import *
from .odesystem import *

__all__ = [s for s in dir() if not s.startswith('_')]

try:
    from numpy.testing import Tester
    test = Tester().test
except:
    #testing could not be loaded, old numpy version
    pass