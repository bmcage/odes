#
# odes - Extra ode integrators
#

from info import __doc__

from dae import *

__all__ = filter(lambda s:not s.startswith('_'),dir())

try:
    from numpy.testing import Tester
    test = Tester().test
except:
    #testing could not be loaded, old numpy version
    pass