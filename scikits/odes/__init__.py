#
# odes - Extra ode integrators
#
from .info import __doc__

from .dae import *
from .ode import *

__all__ = ['test'] + [s for s in dir() if not s.startswith('_')]

try:
    from numpy.testing import Tester
    test = Tester().test
except:
    #testing could not be loaded, old numpy version
    pass