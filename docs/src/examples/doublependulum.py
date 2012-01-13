# Authors: B. Malengier 
"""
Example to show the use of stepwise solving, using the ida or the ddaspk 
solver. It also shows how a class is used to hold the problem data.

This example shows how to solve the double pendulum in full coordinate space.
This results in a dae system.

The problem is easily stated: a first pendulum must move on a circle with 
radius 5 and has mass m1, a second one is attached and must move on a circle 
with radius 2, it has a mass m2, and the gravitational accelleration is g. 
The Lagragian is 

    L = 1/2 m1 (u1^2 + v1^2) - m1 g y1 + 1/2 m2 (u2^2 + v2^2) - m2 g y2
            + lambda1/2 (x1^2+y1^2 - 25) + lambda2/2 ((x2-x1)^2+(y2-y1)^2 - 4)
            
The last two terms are Lagrange multipliers following from the two constraints:
x1^2+y1^2 - 25 = 0 and (x2-x1)^2+(y2-y1)^2 - 4 = 0. 

We arrive at the Euler-Lagrange differential equations for the problem, which 
will have 8 DE and 2 constraints.
We add however two more constraints to the problem, following from deriving to 
time the two known constraints: 

x1*u1+y1*v1 = 0 and (x2-x1)*(u2-u1)^2+(y2-y1)*(v2-v1)=0.

These equations do not change the derived equations if their lagrange multiplier
is supposed independent of time, as the terms they contribute
annihilate each other. 
We introduce the lagrange multipliers back into the residual by adding them 
to the velocity terms as 
   dx1/dt = u1 - lambda3 * x1, ...
We arrive at the stablized index 2 double pendulum model

An alternative is to use an index 1 model, which does not need the stabilizing 
terms. This leads to 8 DE, and 2 algebraic equations. These 2 follow from 
deriving the two contraints twice to time, and replacing the second derivatives
by their definitions.

The solver used approximates the Jacobian. As the Jacobian of the resulting
system is easily computed analytically, we can aid the solver by providing the
Jacobian. This results in the third option: the index 1 problem with Jacobian 
prescribed.

The algorithm first needs to find initial conditions for the derivatives,
then it solves the problem at hand. We take g=9.8, m1=2, m2=1, r1=5, r2=2, 
where the initial position is at rest horizontally.
The solver index 2 problem becomes unstable around 109 seconds. All solvers 
show an increased deviation of the energy invariant from this time onwards.

The algorithm computes the solution over 125 seconds, and plots the orbits of
the pendula, as well as the energy of the system (should be conserved at the
beginning value E=0). Next you can optionally have an animation created of the
solution (this is slow!). For this you need to have the ffmpeg program 
installed. The animation is stored in the directory anidoublependulum.

"""

from __future__ import print_function, division
import numpy as np
from numpy import (arange, zeros, array, sin, cos, asarray, sqrt, pi)
from scikits.odes.sundials.common_defs import ResFunction, JacFunction
from scikits.odes.sundials import ida
from scikits.odes import dae
import pylab
import os

#Set following False to not compute solution with ddaspk
alsoddaspk = True
ATOL = 1e-9
RTOL = 1e-7

#class to hold the problem data
class Doublependulum():
    """ The problem class with the residual function and the constants defined
    """
    
    #default values
    deftend = 125.
    deftstep = 1e-2
    defx0 = 5
    defy0 = 0
    defx1 = 7
    defy1 = 0

    defg = 9.8
    defm1 = 2.0
    defm2 = 1.0
    defradius1 = 5.
    defradius2 = 2.
    
    def __init__(self, data=None, type='index2'):
        self.tend = Doublependulum.deftend
        self.tstep = Doublependulum.deftstep
        self.x0 = Doublependulum.defx0
        self.y0 = Doublependulum.defy0
        self.x1 = Doublependulum.defx1
        self.y1 = Doublependulum.defy1
        self.m1 = Doublependulum.defm1
        self.m2 = Doublependulum.defm2
        self.radius1 = Doublependulum.defradius1
        self.radius2 = Doublependulum.defradius2
        self.g = Doublependulum.defg
        
        #res and jac needed for ddaspk run
        self.res = None
        self.jac = None
        
        if data is not None:
            self.tend = data.deftend
            self.tstep = data.deftstep
            self.x0 = data.x0
            self.y0 = data.y0
            self.x1 = data.x1
            self.y1 = data.y1
            self.m1 = data.m1
            self.m2 = data.m2
            self.radius1 = data.radius1
            self.radius2 = data.radius2
            self.g = data.g
        
        self.stop_t  = arange(.0, self.tend, self.tstep)
        
        lambdaval = 0.0
        if type == 'index2':
            self.neq = 12
            self.z0 =  array([self.x0, self.y0, self.x1, self.y1, 0., 0., 0., 
                              0., lambdaval, lambdaval, lambdaval, lambdaval]) 
            self.zprime0 = array([0., 0., 0., 0., -lambdaval*self.x0,
                                  -lambdaval*self.y0-self.g, -lambdaval*self.x1,
                                  -lambdaval*self.y1-self.g, 0., 0., 0., 0.],
                                  float)
            self.algvar_idx = [8, 9, 10, 11]
            self.algvar = array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1])
            self.exclalg_err = True
        elif type == 'index1' or type == 'index1_jac':
            self.neq = 10
            self.z0 =  array([self.x0, self.y0, self.x1, self.y1, 0., 0., 0.,
                              0., lambdaval, lambdaval]) 
            self.zprime0 = array([0., 0., 0., 0., -lambdaval*self.x0, 
                                  -lambdaval*self.y0-self.g, 
                                  -lambdaval*self.x1, 
                                  -lambdaval*self.y1-self.g, 0., 0.], float)
            self.algvar_idx = [8, 9]
            self.algvar = array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1])
            self.exclalg_err = False

    def set_res(self, resfunction):
        """Function to set the resisual function as required by IDA
           needed for the ddaspk simulation"""
        self.res = resfunction

    def set_jac(self, jacfunction):
        """Function to set the resisual function as required by IDA
           needed for the ddaspk simulation"""
        self.jac = jacfunction
        
    def ddaspk_res(self, tres, yy, yp, idares):
        """the residual function as required by ddaspk"""
        fres = np.empty(self.neq, float)
        self.res.evaluate(tres, yy, yp, fres, None)
        return fres

    def ddaspk_jac(self, tres, yy, yp, cj, jac):
        """the jacobian function as required by ddaspk"""
        self.jac.evaluate(tres, yy, yp, cj, jac)
        return 0

#classes for the equations, as needed for the chosen solution method
class resindex2(ResFunction):
    """ Residual function class as needed by the IDA DAE solver"""
    
    def set_dblpend(self, dblpend):
        """ Set the double pendulum problem to solve to have access to
            the data """
        self.dblpend = dblpend

    def evaluate(self, t, x, xdot, result, userdata):
        """ compute the residual of the ode/dae system"""
        m1 = self.dblpend.m1
        m2 = self.dblpend.m2
        g = self.dblpend.g
        result[0]= m1*xdot[4]        -x[9]*(x[0] - x[2])  - x[0]*x[8]
        result[1]= m1*xdot[5] + g*m1 -x[9]*(x[1] - x[3])  - x[1]*x[8]
        result[2]= m2*xdot[6]       + x[9]*(x[0] - x[2])
        result[3]= m2*xdot[7] +g*m2 + x[9]*(x[1] - x[3])
        result[4]= xdot[0] - x[4] + x[10]*x[0]
        result[5]= xdot[1] - x[5] + x[10]*x[1]
        result[6]= xdot[2] - x[6] + x[11]*x[2]
        result[7]= xdot[3] - x[7] + x[11]*x[3]
        result[8]= self.dblpend.radius1*self.dblpend.radius1 \
                    - x[0]**2 - x[1]**2
        result[9]= self.dblpend.radius2*self.dblpend.radius2 \
                    - (x[0] - x[2])**2 - (x[1] - x[3])**2
        result[10]= x[0]*x[4] + x[1]*x[5]
        result[11]=(x[4] - x[6])*(x[2] - x[0]) - (x[1] - x[3])*(x[5] - x[7])
        return 0

class resindex1(ResFunction):
    """ Residual function class as needed by the IDA DAE solver"""
    
    def set_dblpend(self, dblpend):
        """ Set the double pendulum problem to solve to have access to
            the data """
        self.dblpend = dblpend

    def evaluate(self, tres, yy, yp, result, userdata):
        m1 = self.dblpend.m1
        m2 = self.dblpend.m2
        g = self.dblpend.g
        
        result[0]= m1*yp[4]        - yy[9]*(yy[0] - yy[2])  - yy[0]*yy[8]
        result[1]= m1*yp[5] + g*m1 - yy[9]*(yy[1] - yy[3])  - yy[1]*yy[8] 
        result[2]= m2*yp[6]        + yy[9]*(yy[0] - yy[2]) 
        result[3]= m2*yp[7] + g*m2 + yy[9]*(yy[1] - yy[3]) 
        result[4]= yp[0] - yy[4] #+ yy[10]*yy[0]
        result[5]= yp[1] - yy[5] #+ yy[10]*yy[1]
        result[6]= yp[2] - yy[6] #+ yy[11]*yy[2]
        result[7]= yp[3] - yy[7] #+ yy[11]*yy[3]
        #result[8]= Doublependulum.radius1*Doublependulum.radius1 \
        #            - yy[0]**2 - yy[1]**2
        #result[9]= Doublependulum.radius2*Doublependulum.radius2 \
        #            - (yy[0] - yy[2])**2 - (yy[1] - yy[3])**2
        #result[8]= yy[0]*yy[4] + yy[1]*yy[5]
        #result[9]=(yy[4] - yy[6])*(yy[2] - yy[0]) - (yy[1] - yy[3])*(yy[5] - yy[7])
        result[8] = yy[4]**2 + yy[5]**2 + yy[8]/m1*(yy[0]**2 + yy[1]**2) \
                    - g * yy[1] + yy[9]/m1 *(yy[0]*(yy[0]-yy[2]) +
                                            yy[1]*(yy[1]-yy[3]) )
        result[9] = (yy[4]-yy[6])**2 + (yy[5]-yy[7])**2 \
                  + yy[9]*(1./m1+1./m2)*((yy[0]-yy[2])**2 + (yy[1]-yy[3])**2)\
                  + yy[8]/m1 *(yy[0]*(yy[0]-yy[2]) + yy[1]*(yy[1]-yy[3]) )
        return 0

class jacindex1(JacFunction):
    
    def set_dblpend(self, dblpend):
        """ Set the double pendulum problem to solve to have access to
            the data """
        self.dblpend = dblpend

    def evaluate(self, tres, yy, yp, cj, jac):
        
        m1 = self.dblpend.m1
        m2 = self.dblpend.m2
        g = self.dblpend.g
        for i in range(10):
            jac[i][:] = 0.
        jac[0][0] = - yy[9]   - yy[8] 
        jac[0][2] =  yy[9]
        jac[0][4] = cj * m1
        jac[0][8] = - yy[0]
        jac[0][9] = - (yy[0] - yy[2])
        jac[1][1] = - yy[9] - yy[8]
        jac[1][3] = yy[9]
        jac[1][5] = cj * m1
        jac[1][8] = - yy[1]
        jac[1][9] = - (yy[1] - yy[3])
        jac[2][0] = yy[9]
        jac[2][2] = -yy[9]
        jac[2][6] = cj * m2
        jac[2][9] = (yy[0] - yy[2])
        jac[3][1] = yy[9]
        jac[3][3] = -yy[9]
        jac[3][7] = cj * m2
        jac[3][9] = (yy[1] - yy[3])
        jac[4][0] = cj
        jac[4][4] = -1
        jac[5][1] = cj
        jac[5][5] = -1
        jac[6][2] = cj
        jac[6][6] = -1
        jac[7][3] = cj
        jac[7][7] = -1
        jac[8][0] = (yy[8]+yy[9])/m1*2*yy[0] - yy[9]/m1 * yy[2] 
        jac[8][1] = (yy[8]+yy[9])/m1*2*yy[1] - yy[9]/m1 * yy[3] - g
        jac[8][2] = - yy[9]/m1 * yy[0]
        jac[8][3] = - yy[9]/m1 * yy[1]
        jac[8][4] = 2*yy[4] 
        jac[8][5] = 2*yy[5] 
        jac[8][8] = 1./m1*(yy[0]**2 + yy[1]**2)
        jac[8][9] = 1./m1 *(yy[0]*(yy[0]-yy[2]) + yy[1]*(yy[1]-yy[3]) )
        jac[9][0] = yy[9]*(1./m1+1./m2)*2*(yy[0]-yy[2]) + \
                    yy[8]/m1 *(2*yy[0] - yy[2]) 
        jac[9][1] = yy[9]*(1./m1+1./m2)*2*(yy[1]-yy[3]) + \
                    yy[8]/m1 *(2*yy[1] - yy[3]) 
        jac[9][2] = - yy[9]*(1./m1+1./m2)*2*(yy[0]-yy[2]) - \
                    yy[8]/m1 * yy[0]
        jac[9][3] = - yy[9]*(1./m1+1./m2)*2*(yy[1]-yy[3]) 
        jac[9][4] = 2*(yy[4]-yy[6]) 
        jac[9][5] = 2*(yy[5]-yy[7]) 
        jac[9][6] = -2*(yy[4]-yy[6])
        jac[9][7] = -2*(yy[5]-yy[7])
        jac[9][8] = 1./m1 *(yy[0]*(yy[0]-yy[2]) + yy[1]*(yy[1]-yy[3]) )
        jac[9][9] = (1./m1+1./m2)*((yy[0]-yy[2])**2 + (yy[1]-yy[3])**2)

#Now that all is defined, solve it
def main():
    """
    The main program: instantiate a problem, then use odes package to solve it
    """
    input = raw_input("Solve as\n 1 = index 2 problem\n 2 = index 1 problem\n"
                " 3 = index 1 problem with jacobian\n 4 = info\n\n"
                "Answer (1,2,3 or 4) : ")
    jac = None
    if input == '1':
        problem = Doublependulum(type='index2')
        res = resindex2()
    elif input == '2':
        problem = Doublependulum(type='index1')
        res = resindex1()
    elif input == '3':
        problem = Doublependulum(type='index1_jac')
        res = resindex1()
        jac = jacindex1()
    else:
        print(__doc__)
        return
    
    z = [0]*(1+len(problem.stop_t)); zprime = [0]*(1+len(problem.stop_t))

    res.set_dblpend(problem)
    if jac:
        jac.set_dblpend(problem)

    solver = ida.IDA(res,
                compute_initcond='yp0',
                first_step=1e-18,
                atol=ATOL,rtol=RTOL,
                jacfn=jac,
                algebraic_vars_idx=problem.algvar_idx,
                exclude_algvar_from_error=problem.exclalg_err,
                )

    z[0] = np.empty(problem.neq, float)
    zprime[0] = np.empty(problem.neq, float)
    t0_init = solver.init_step(0., problem.z0, problem.zprime0, z[0], zprime[0])
    print ('init time', t0_init)
    realtime = [t0_init]
    #flag, rt = solver.step(t0_init, z[0], zprime[0])

    i=1
    error = False
    for time in problem.stop_t[1:]:
            #print 'at time', time
            z[i] = np.empty(problem.neq, float)
            zprime[i] = np.empty(problem.neq, float)
            flag, rt = solver.step(time, z[i], zprime[i])
            realtime += [rt]
            #print 'sol at ', time, z[i]
            
            i += 1
            if flag != 0:
                error = True
                print('Error in solver, breaking solution at time %g' % time)
                break

    fres = np.empty(problem.neq, float)
    res.evaluate(problem.stop_t[i-1], z[i-1], zprime[i-1], fres, None)
    print('last sol has residual: ', fres)

    nr = i
    x1t = asarray([z[i][0] for i in range(nr)])
    y1t = asarray([z[i][1] for i in range(nr)])
    x2t = asarray([z[i][2] for i in range(nr)])
    y2t = asarray([z[i][3] for i in range(nr)])
    xp1t = asarray([z[i][4] for i in range(nr)])
    yp1t = asarray([z[i][5] for i in range(nr)])
    xp2t = asarray([z[i][6] for i in range(nr)])
    yp2t = asarray([z[i][7] for i in range(nr)])
    energy = problem.m1*problem.g*y1t + \
                problem.m2*problem.g*y2t + \
                .5 *(problem.m1 * (xp1t**2 + yp1t**2) 
                     + problem.m2 * (xp2t**2 + yp2t**2) )
    initenergy = energy[0]

    #solve the same with ddaspk
    #sys.exit()
    if alsoddaspk:
        ddaspkz = [0]*(1+len(problem.stop_t))
        ddaspkzprime = [0]*(1+len(problem.stop_t))
        ddaspkz[0] = np.empty(problem.neq, float)
        ddaspkzprime[0] = np.empty(problem.neq, float)
        
        problem.set_res(res)
        if jac:
            problem.set_jac(jac)
            ig = dae(problem.ddaspk_res, problem.ddaspk_jac)
        else:
            ig = dae(problem.ddaspk_res, None)
        #first compute the correct initial condition from the values of z0
        ig.set_integrator('ddaspk',algebraic_var=problem.algvar,
                            compute_initcond='yode0',
                            first_step=1e-18,
                            atol=ATOL,rtol=RTOL)
        ig.set_initial_value(problem.z0, problem.zprime0,  t=0.0)

        i=0
        ddaspkz[i],  ddaspkzprime[i] = ig.solve(1e-18);
        assert ig.successful(), (problem,)
        print('ddaspk started from z0 = ', problem.z0)
        print('ddaspk initial condition calculated, [z,zprime] = [', z[0], zprime[0], ']')

        ig.set_integrator('ddaspk',algebraic_var=problem.algvar,
                            first_step=1e-9,
                            atol=1e-6,rtol=0.5e-5, #atol=1e-8,rtol=1e-8,
                            exclude_algvar_from_error=problem.exclalg_err,
                            nsteps = 1500)
        ig.set_initial_value(ddaspkz[0], ddaspkzprime[0], t=0.0)
        i=1
        error = False
        for time in problem.stop_t[1:]:
                #print 'at time', time
                ddaspkz[i],  ddaspkzprime[i] = ig.solve(time)
                #print 'sol at ', time, z[i]
                
                i += 1
                if not ig.successful():
                    error = True
                    print('Error in ddaspk solver, breaking solution at time %g' % time)
                    break
        dnr = i
        ddaspkx1t = asarray([ddaspkz[i][0] for i in range(dnr)])
        ddaspky1t = asarray([ddaspkz[i][1] for i in range(dnr)])
        ddaspkx2t = asarray([ddaspkz[i][2] for i in range(dnr)])
        ddaspky2t = asarray([ddaspkz[i][3] for i in range(dnr)])
        ddaspkxp1t = asarray([ddaspkzprime[i][0] for i in range(dnr)])
        ddaspkyp1t = asarray([ddaspkzprime[i][1] for i in range(dnr)])
        ddaspkxp2t = asarray([ddaspkzprime[i][2] for i in range(dnr)])
        ddaspkyp2t = asarray([ddaspkzprime[i][3] for i in range(dnr)])
        ddaspkenergy = problem.m1*problem.g*ddaspky1t + \
                    problem.m2*problem.g*ddaspky2t + \
                    .5 *(problem.m1 * (ddaspkxp1t**2 + ddaspkyp1t**2) 
                         + problem.m2 * (ddaspkxp2t**2 + ddaspkyp2t**2) )
        ddaspkrealtime = problem.stop_t[:dnr]
    pylab.ion()
    pylab.figure(1)
    pylab.subplot(211)
    pylab.title('IDA solution option %s' % input)
    pylab.scatter(x1t, y1t)
    pylab.scatter(x2t, y2t)
    pylab.xlim(-10, 10)
    pylab.ylim(-8, 2)
    pylab.axis('equal')
    pylab.subplot(212)
    pylab.plot(realtime, energy, 'b')
    pylab.title('Energy Invariant Violation for a Double Pendulum')
    pylab.xlabel('Time (s)')
    pylab.ylabel('Total Energy')
    pylab.axis()
    pylab.show()
    if alsoddaspk:
        pylab.ion()
        pylab.figure(2)
        pylab.subplot(211)
        pylab.title('DDASPK solution option %s' % input)
        pylab.scatter(ddaspkx1t, ddaspky1t)
        pylab.scatter(ddaspkx2t, ddaspky2t)
        pylab.xlim(-10, 10)
        pylab.ylim(-8, 2)
        pylab.axis('equal')
        pylab.subplot(212)
        pylab.plot(ddaspkrealtime, ddaspkenergy, 'b')
        pylab.title('Energy Invariant Violation for a Double Pendulum')
        pylab.xlabel('Time (s)')
        pylab.ylabel('Total Energy DDASPK solution')
        pylab.axis()
        pylab.show()

    def circle(ax, x, y, r, color='r'):
        count = 20
        ax.fill([x + r * cos(2.* i * pi/count) for i in range(count+1)],
                [y + r * sin(2.* i * pi/count) for i in range(count+1)],
                color)

    def line(ax, x1, y1, x2, y2, color='black'):
        norm = sqrt((y2-y1)**2 + (x2-x1)**2)
        normal = [(y2-y1)/norm, -(x2-x1)/norm]
        ax.fill(
        [x1-0.01*normal[0],x1+0.01*normal[0],x2+0.01*normal[0],x2-0.01*normal[0]],
        [y1-0.01*normal[1],y1+0.01*normal[1],y2+0.01*normal[1],y2-0.01*normal[1]],
                color)

    def draw(ax, nr):
        sol = z[nr]
        line(ax, 0., 0., sol[0], sol[1])
        line(ax, sol[0], sol[1], sol[2], sol[3])
        circle(ax, sol[0], sol[1], 0.5)
        circle(ax, sol[2], sol[3], 0.5)

    def drawonesol(nr, sizex, sizey, ext=None):
        pylab.clf()
        a = pylab.axes()
        draw(a, nr)
        pylab.axis('scaled')
        pylab.xlim(-sizex, sizex)
        pylab.ylim(-sizey, sizey)
        if ext is None:
            ext = nr
        pylab.savefig('figsdoublependulum' + os.sep + 'outsol%08i.png' % ext)

    def open_file_with_default_application( file_path ):
        """
        Launch a program to open an arbitrary file. The file will be opened using 
        whatever program is configured on the host as the default program for that 
        type of file.
        """
        
        norm_path = os.path.normpath( file_path )
        
        if not os.path.exists(norm_path):
            print("%s does not exist" % file_path)
            return
            
        if os.sys.platform == 'win32':
            try:
                os.startfile(norm_path)
            except WindowsError as msg:
                print("Error Opening File. " + str(msg))
        else:
            search = os.environ['PATH'].split(':')
            for path in search:
                prog = os.path.join(path, 'xdg-open')
                if os.path.isfile(prog):
                    os.spawnvpe(os.P_NOWAIT, prog, [prog, norm_path], os.environ)
                    return

    def create_animation(sizex, sizey, ext):
        """
        The calculation step is 1e-2, so output every 5 solutions or 0.05, means 
        a frame rate of 20 frames per second
        """
        import shutil
        
        fps = 20
        if os.path.isdir('figsdoublependulum'):
            shutil.rmtree('figsdoublependulum')
        os.mkdir('figsdoublependulum')
        if not os.path.isdir('anidoublependulum'):
            os.mkdir('anidoublependulum')
        
        pylab.figure(2)
        secs = 0
        frame = 0
        print('Generating output ...\n')
        for solnr in range(0,nr,5):
            drawonesol(solnr, sizex, sizey, frame)
            frame += 1
            if solnr // 500 != secs :
                secs = solnr // 500
                print('     ... at %i seconds ' % (secs * 5 )) 
        
        print('Creating movie using ffmpeg with output ... \n')
        import subprocess
        subprocess.call(['ffmpeg', '-r', '20', '-i', 'figsdoublependulum' + os.sep + 
                        'outsol%8d.png',  '-f',  'avi', '-vcodec', 'mpeg2video', '-y', 
                        'anidoublependulum' + os.sep + 
                                            'doublependulum'+ext+'.mpg'])
        #remove unused pictures
        shutil.rmtree('figsdoublependulum')
        #opening movie with default player
        print('Opening user with default application ... \n')
        open_file_with_default_application('anidoublependulum' + os.sep +
                    'doublependulum'+ext+'.mpg')

    input2 = raw_input('Create animation of the solution? (y/n): ')
    print('\n')
    if (input2 == 'y' or input2 == 'yes'):
        extend = problem.radius1 + problem.radius2 + 1
        create_animation(extend, extend, input)

if __name__ == "__main__":
    main()
