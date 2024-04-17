---
title: 'ODES: a high level interface to ODE and DAE solvers'
tags:
  - ode
  - dae
  - sundials
  - root finding
authors:
 - name: Benny Malengier
   orcid: 0000-0001-8383-8068
   affiliation: 1
 - name: Pavol Kišon
   orcid: 
   affiliation: 2
 - name: James Tocknell
   orcid: 0000-0001-6637-6922
   affiliation: 3
 - name: Claas Abert
   orcid: 0000-0002-4999-0311
   affiliation: 4
 - name: Florian Bruckner
   orcid: 0000-0001-7778-6855
   affiliation: 5
 - name: Marc-Antonio Bisotti
   orcid:
   affiliation: 6
affiliations:
 - name: Ghent University
   index: 1
 - name: Ghent University
   index: 2
 - name: Macquarie University
   index: 3
 - name: University of Vienna
   index: 4
 - name: University of Vienna
   index: 5
 - name: University of Southampton
   index: 6
date: 06 January 2017
bibliography: paper.bib
---

# Summary

ODES is a scikit offering extra ODE/DAE solvers, as an extension to what is available in SciPy [@scipy2001], with a high level interface.

ODES offers a high level object oriented API to Differential Equation solving in Python. The backbone of ODES is the [SUNDIALS package](http://computation.llnl.gov/projects/sundials) [@hindmarsh2005sundials], which offers ODE and DAE solvers with root finding, preconditioning, error control and more.
ODES can be used in Python 2.7 or 3.3-3.6, while for speed purposes the integrator can be a Cython function instead of a pure Python function. Comparison between different methods for a test problem is given in following graph:
![Performance graph](./PerformanceTests.png)
You can generate above graph via the [Performance notebook](https://github.com/bmcage/odes/blob/master/ipython_examples/Performance%20tests.ipynb).

As ODES is a Python package, much thought has been given on ease of use. On one hand, novices can use the *odeint* convenience function, see example use in [simple.py](https://github.com/bmcage/odes/blob/master/docs/examples/odeint/simple.py). On the other hand, an object oriented interface is available via the *ode* and *dae* objects. See the [Documentation](https://scikits-odes.readthedocs.io/en/latest/) for details.

For users new to solving ODEs, we recommend reading through @Hairer1993, which
contains useful advice to find successful solutions.

# References
  
