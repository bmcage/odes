## Automatically adapted for scipy Oct 21, 2005 by

"""
ODE and DAE routines
====================
   dae          -- Solve DAE systems
   ode         -- Solve ODE systems, an extension to the scipy.integrate API

 Interface to numerical integrators of DAE/ODE systems.

 * Via the dae class:

   ida           -- The sundials IDA general dae solver. BDF method or Adams
                    method, ideal for stiff problems
   ddaspk        -- General dae solver in Fortran present for hystorical and
                    comparison reasons (use ida instead)
   lsodi         -- General dae solver in Fortran present for hystorical and
                    comparison reasons (use ida instead)

* Via the ode class:

   cvode         -- The sundials CVODE general ode solver. BDF method or Adams
                    method, ideal for stiff problems
   dopri5        -- Runge Kutta solver, ideal for non-stiff problems


Note: as part of scipy.integrate:
=================================
   odeint        -- General integration of ordinary differential equations.
   ode           -- Integrate ode

"""

postpone_import = 1
