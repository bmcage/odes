# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:11:17 2016

@author: benny

Example of using the odeint convenience method
"""
import numpy as np

#   The second order differential equation for the angle `theta` of a
#   pendulum acted on by gravity with friction can be written::
b = 0.25
c = 5.0
def pend(t, y, out):
    theta, omega = y
    out[:] = [omega, -b*omega - c*np.sin(theta)]

y0 = [np.pi - 0.1, 0.0]
t = np.linspace(0, 10, 101)

from scikits.odes.odeint import odeint
sol = odeint(pend, t, y0)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(sol.values.t, sol.values.y[:, 0], 'b', label='$\\theta(t)$')
plt.plot(sol.values.t, sol.values.y[:, 1], 'g', label='$\\omega(t)$')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
#plt.show()

# now the same with a class method
class pendulum(object):
    def __init__(self, b=0.25, c=5.0):
        self.b = b
        self.c = c

    def rhs(self, t, y, out):
        theta, omega = y
        out[:] = [omega, -b*omega - c*np.sin(theta)]

# we see difference in terms of b parameter
plt.figure()
sols = [(b, odeint(pendulum(b=b).rhs, t, y0)) for b in np.linspace(0.,1.,11)]
[plt.plot(sol.values.t, sol.values.y[:, 0], label='b={}'.format(b)) \
        for (b, sol) in sols]
plt.legend(loc='best')
plt.xlabel('t')
plt.title('Change in $\\theta(t)$ in terms of b')
plt.grid()

plt.figure()
test = pendulum(b=0.25)
sols = [(method, odeint(test.rhs, t, y0, method=method)) \
            for method in ['bdf', 'admo', 'rk5', 'rk8', 'beuler', 'trapz']]
[plt.plot(sol.values.t, sol.values.y[:, 0], label='method={}'.format(method)) \
            for (method, sol) in sols]
plt.legend(loc='best')
plt.xlabel('t')
plt.title('Differences between ODE methods')
plt.grid()

plt.show()  
        