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
   orcid: 
   affiliation: 3
 - name: Claas Abert
   orcid: 
   affiliation: 4
 - name: Florian Bruckner
   orcid: 
   affiliation: 5
affiliations:
 - name: Ghent University
   index: 1
 - name: Ghent University
   index: 2
 - name: Macquarie University
   index: 3
date: 06 January 2017
bibliography: paper.bib
---

# Summary

ODES is a scikit offering extra ode/dae solvers, as an extension to what is available in scipy, with a high level interface.

Odes offers a high level object oriented API to Differential Equation solving in python. The backbone of odes is the [SUNDIALS package](http://computation.llnl.gov/projects/sundials), which offers ODE and DAE solvers with root finding, preconditioning, error control and more.
Odes can be used in python 2.7 or 3.2, while for speed purposes the integrator can be a cython function in stead of a pure python function. Comparison between different methods for a test problem is given in following graph:
![Performance graph](https://github.com/bmcage/odes/blob/master/docs/ipython/Performance%20tests.png)
You can generate above graph via the [Performance notebook](https://github.com/bmcage/odes/blob/master/docs/ipython/Performance%20tests.ipynb).

As odes is a python package, much thought has been given on ease of use. On one hand, novices can use the *odeint* convenience function, see example use in [simple.py](https://github.com/bmcage/odes/blob/master/docs/src/examples/odeint/simple.py). On the other hand, an object oriented interface is available via the *ode* and *dae* objects. See the Documentation for details.

# References
  
