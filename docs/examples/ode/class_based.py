# Authors: B. Malengier
"""
This example shows the most simple way of using a solver.
We solve free vibration of a simple oscillator::
        m \ddot{u} + k u = 0, u(0) = u_0, \dot{u}(0) = \dot{u}_0
using the CVODE solver, which means we use a rhs function of \dot{u}.
Solution::
        u(t) = u_0*cos(sqrt(k/m)*t)+\dot{u}_0*sin(sqrt(k/m)*t)/sqrt(k/m)

"""
from __future__ import print_function
from numpy import asarray, cos, sin, sqrt
import numpy as np
from scikits.odes.sundials.cvode import CVODE, StatusEnum, CV_WrapJacRhsFunction
from collections import namedtuple

#data
k = 4.0
m = 1.0
t1 = 10.
#initial data on t=0, x[0] = u, x[1] = \dot{u}, xp = \dot{x}
initx = [1, 0.1]

def rhseqn(t, x):
    """ we create rhs equations for the problem"""
    return [
        x[1],
        - k/m * x[0]
    ]

def jaceqn(t, x,):
    jac = np.zeros((2,2))
    jac[0,1] = 1
    jac[1,0] = -k/m
    return jac


def rootfn(t, x):
    return (
        x[0],
        x[1],
        t - t1,
        np.sin(t),
    )

Root = namedtuple("Root", ["index", "rootsfound"])
Results = namedtuple("Results", ["t", "x", "e"],)

class System:
    def __init__(self, dots, jac, events, num_events):
        self._dots = dots
        self._jac = jac
        self._events = events
        self.num_events = num_events
        self.solver = CVODE(
            self.dots,
            jacfn=self.jac,
            old_api=False,
            one_step_compute=True,
            rootfn=self.events,
            nr_rootfns=self.num_events,
        )

    def dots(self, t, x, xdot,):# userdata=None,):
        xdot[:] = self._dots(t, x)

    def jac(self, t, x, xdot, jac, userdata=None,):
        jac[...] = self._jac(t, x)

    def events(self, t, x, g):
        g[:] = self._events(t, x)

    def simulate(self, t0, x0, tf, results=None):

        if results is None:
            results = Results([],[],[])

        dense_t = results.t
        dense_y = results.x
        roots = results.e

        solver = self.solver

        solver.init_step(t0, x0)
        solver.set_options(tstop=tf)

        dense_t.append(np.copy(t0))
        dense_y.append(np.copy(x0))

        for cnt in range(1000):
            res = solver.step(t1)
            print(cnt, res.flag, res.values.t)
            dense_t.append(np.copy(res.values.t))
            dense_y.append(np.copy(res.values.y))
            match res.flag:
                case StatusEnum.ROOT_RETURN:
                    rootsfound = solver.rootinfo()
                    roots.append(Root(cnt, rootsfound))

                    if res.values.t == tf:
                        break

                case StatusEnum.TSTOP_RETURN:
                    #continue
                    break

        return results



sys = System(rhseqn, jaceqn, rootfn, 4)
res1 = sys.simulate(0., initx, 11.)
res2 = sys.simulate(0., initx, 10.)
