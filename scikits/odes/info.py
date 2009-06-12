## Automatically adapted for scipy Oct 21, 2005 by

"""
ODE and DAE routines
====================
   dae          -- Solve DAE systems, similar to the scipy.integrate ode 
                   class
   odesystem    -- Solve ODE systems, an extension to the scipy.integrate ode 
                   class

 Interface to numerical integrators of DAE/ODE systems.

 * Via the dae class:

   odesIDA       -- The sundials IDA general dae solver
   ddaspk        -- General dae solver
   lsodi         -- General dae solver

* Via the odesystem class:
 
   odesCVODE     -- The sundials CVODE general ode solver
   scipy.integrate ode solvers are available too (eg Fortran VODE/ZVODE)

Note: as part of scipy.integrate:
=================================
   odeint        -- General integration of ordinary differential equations.
   ode           -- Integrate ODE using VODE and ZVODE routines.

"""

postpone_import = 1
