# Authors: B. Malengier 
"""
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
from numpy import (arange, zeros, array, sin, cos, asarray, sqrt, pi)
from scikits.odes import dae
import pylab
import os

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
        
        self.stop_t  = arange(.0, self.tend, self.tstep)[1:]
        
        lambdaval = 0.0
        if type == 'index2':
            self.neq = 12
            self.z0 =  array([self.x0, self.y0, self.x1, self.y1, 0., 0., 0., 0., lambdaval,
                                 lambdaval, lambdaval, lambdaval]) 
            self.zprime0 = array([0., 0., 0., 0., -lambdaval*self.x0, -lambdaval*self.y0-self.g
                            , -lambdaval*self.x1, -lambdaval*self.y1-self.g, 0., 0., 0., 0.], float)
            self.res = self.resindex2
            self.algvar = array([1,1,1,1,1,1,1,1,-1,-1,-1,-1])
            self.exclalg_err = True
        elif type == 'index1' or type == 'index1_jac':
            self.neq = 10
            self.z0 =  array([self.x0, self.y0, self.x1, self.y1, 0., 0., 0., 0., lambdaval,
                                 lambdaval]) 
            self.zprime0 = array([0., 0., 0., 0., -lambdaval*self.x0, -lambdaval*self.y0-self.g
                            , -lambdaval*self.x1, -lambdaval*self.y1-self.g, 0., 0.], float)
            self.res = self.resindex1
            self.algvar = array([1,1,1,1,1,1,1,1,-1,-1])
            self.exclalg_err = False
            if type == 'index1_jac':
                self.jac = self.jacindex1
    
    def resindex2(self, tres, yy, yp):
        m1 = self.m1
        m2 = self.m2
        g = self.g
        tmp = zeros(self.neq)
        tmp[0]= m1*yp[4]        -yy[9]*(yy[0] - yy[2])  - yy[0]*yy[8]
        tmp[1]= m1*yp[5] + g*m1 -yy[9]*(yy[1] - yy[3])  - yy[1]*yy[8]
        tmp[2]= m2*yp[6]       + yy[9]*(yy[0] - yy[2])
        tmp[3]= m2*yp[7] +g*m2 + yy[9]*(yy[1] - yy[3])
        tmp[4]= yp[0] - yy[4] + yy[10]*yy[0]
        tmp[5]= yp[1] - yy[5] + yy[10]*yy[1]
        tmp[6]= yp[2] - yy[6] + yy[11]*yy[2]
        tmp[7]= yp[3] - yy[7] + yy[11]*yy[3]
        tmp[8]= self.radius1*self.radius1 \
                    - yy[0]**2 - yy[1]**2
        tmp[9]= self.radius2*self.radius2 \
                    - (yy[0] - yy[2])**2 - (yy[1] - yy[3])**2
        tmp[10]= yy[0]*yy[4] + yy[1]*yy[5]
        tmp[11]=(yy[4] - yy[6])*(yy[2] - yy[0]) - (yy[1] - yy[3])*(yy[5] - yy[7])
        return tmp
    
    def resindex1(self, tres, yy, yp):
        m1 = self.m1
        m2 = self.m2
        g = self.g
        tmp = zeros(self.neq)
        tmp[0]= m1*yp[4]        - yy[9]*(yy[0] - yy[2])  - yy[0]*yy[8]
        tmp[1]= m1*yp[5] + g*m1 - yy[9]*(yy[1] - yy[3])  - yy[1]*yy[8] 
        tmp[2]= m2*yp[6]        + yy[9]*(yy[0] - yy[2]) 
        tmp[3]= m2*yp[7] + g*m2 + yy[9]*(yy[1] - yy[3]) 
        tmp[4]= yp[0] - yy[4] #+ yy[10]*yy[0]
        tmp[5]= yp[1] - yy[5] #+ yy[10]*yy[1]
        tmp[6]= yp[2] - yy[6] #+ yy[11]*yy[2]
        tmp[7]= yp[3] - yy[7] #+ yy[11]*yy[3]
        #tmp[8]= Doublependulum.radius1*Doublependulum.radius1 \
        #            - yy[0]**2 - yy[1]**2
        #tmp[9]= Doublependulum.radius2*Doublependulum.radius2 \
        #            - (yy[0] - yy[2])**2 - (yy[1] - yy[3])**2
        #tmp[8]= yy[0]*yy[4] + yy[1]*yy[5]
        #tmp[9]=(yy[4] - yy[6])*(yy[2] - yy[0]) - (yy[1] - yy[3])*(yy[5] - yy[7])
        tmp[8] = yy[4]**2 + yy[5]**2 + yy[8]/m1*(yy[0]**2 + yy[1]**2) \
                    - g * yy[1] + yy[9]/m1 *(yy[0]*(yy[0]-yy[2]) +
                                            yy[1]*(yy[1]-yy[3]) ) 
        tmp[9] = (yy[4]-yy[6])**2 + (yy[5]-yy[7])**2 \
                  + yy[9]*(1./m1+1./m2)*((yy[0]-yy[2])**2 + (yy[1]-yy[3])**2)\
                  + yy[8]/m1 *(yy[0]*(yy[0]-yy[2]) + yy[1]*(yy[1]-yy[3]) )
        return tmp

    def jacindex1(self, tres, yy, yp, cj, jac):
        m1 = self.m1
        m2 = self.m2
        g = self.g
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

def main():
    """
    The main program: instantiate a problem, then use odes package to solve it
    """
    input = input("Solve as\n 1 = index 2 problem\n 2 = index 1 problem\n"
                " 3 = index 1 problem with jacobian\n 4 = info\n\n"
                "Answer (1,2,3 or 4) : ")
    if input == '1':
        problem = Doublependulum(type='index2')
    elif input == '2':
        problem = Doublependulum(type='index1')
    elif input == '3':
        problem = Doublependulum(type='index1_jac')
    else:
        print(__doc__)
        return
    
    z = [0]*(1+len(problem.stop_t)); zprime = [0]*(1+len(problem.stop_t))

    ig = dae(problem.res, problem.jac)
    #first compute the correct initial condition from the values of z0
    ig.set_integrator('odesIDA',algebraic_var=problem.algvar,
                        compute_initcond='yode0',
                        first_step=1e-9,
                        atol=1e-6,rtol=1e-6)
    ig.set_initial_value(problem.z0, problem.zprime0,  t=0.0)

    i=0
    z[i],  zprime[i] = ig.solve(1e-18);
    assert ig.successful(), (problem,)
    print('started from z0 = ', problem.z0)
    print('initial condition calculated, [z,zprime] = [', z[0], zprime[0], ']')

    ig.set_integrator('odesIDA',algebraic_var=problem.algvar,
                        first_step=1e-9,
                        atol=1e-8,
                        rtol=1e-8,
                        exclude_algvar_from_error=problem.exclalg_err,
                        nsteps = 1500)
    ig.set_initial_value(z[0], zprime[0], t=0.0)

    i=1

    error = False
    for time in problem.stop_t:
            #print 'at time', time
            z[i],  zprime[i] = ig.solve(time)
            #print 'sol at ', time, z[i]
            i += 1
            if not ig.successful():
                error = True
                print('Error in solver, breaking solution at time %g' % time)
                break


    print('last sol', z[i-1], zprime[i-1])
    print('has residual: ', problem.res(problem.stop_t[i-2], z[i-1], 
                                        zprime[i-1]))

    nr = i
    x1t = [z[i][0] for i in range(nr)]
    y1t = [z[i][1] for i in range(nr)]
    x2t = [z[i][2] for i in range(nr)]
    y2t = [z[i][3] for i in range(nr)]
    energy = asarray([problem.m1*problem.g*z[i][1] + \
                problem.m2*problem.g *z[i][3] + \
                .5 *(problem.m1 * (z[i][4]**2 + z[i][5]**2) 
                     + problem.m2 * (z[i][6]**2 + z[i][7]**2) ) for i in 
                        range(nr)])
    initenergy = energy[0]
    time = zeros(nr,float)
    time[0] = 0.0
    if error:
        time[1:]  = problem.stop_t[:nr-1] 
    else:
        time[1:]  = problem.stop_t[:nr]
        
    pylab.figure(1)
    pylab.subplot(211)
    pylab.scatter(x1t, y1t)
    pylab.scatter(x2t, y2t)
    pylab.axis('equal')
    pylab.subplot(212)
    pylab.plot(time, energy, 'b')
    pylab.title('Energy Invariant Violation for a Double Pendulum')
    pylab.xlabel('Time (s)')
    pylab.ylabel('Total Energy')
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
                        'outsol%8d.png',  '-f',  'avi', '-vcodec', 'xvid', '-y', 
                        'anidoublependulum' + os.sep + 
                                            'doublependulum'+ext+'.mpg'])
        #remove unused pictures
        shutil.rmtree('figsdoublependulum')
        #opening movie with default player
        print('Opening user with default application ... \n')
        open_file_with_default_application('anidoublependulum' + os.sep +
                    'doublependulum'+ext+'.mpg')

    input2 = input('Create animation of the solution? (y/n): ')
    print('\n')
    if (input2 == 'y' or input2 == 'yes'):
        extend = problem.radius1 + problem.radius2 + 1
        create_animation(extend, extend, input)

if __name__ == "__main__":
    main()
