# Authors: B. Malengier 
"""
This example shows how to solve the sliding pendulum in full coordinate space.
This results in a dae system.

The problem is easily stated: 
A pendulum can slide with friction along a curve in a plane. We take the
curve

    y = x^2+1/3 cos(\omega x)

There is a mass M on the top part (on the curve), and a mass m at the lower
part, which is distance l away from M.
The Lagrange equations can be written with constraint for the upper end of the
pendulum (must be on the curve, \lambda as Lagrange multiplier), and 
adding also friction with constant k.

The equations contain (x, y, \theta, xdot, ydot, thetadot, \lambda) as unknowns, where
theta is the angle with the vertical y axis.

 \dot{x} = xdot
 \dot{y} = ydot
 \dot{\theta} = thetadot
 (M+m)*\dot{xdot} + m*l*cos(\theta)*\dot{thetadot} = m*l*sin(theta)*thetadot^2 
        - \lambda * (-2*x + \omega/3*sin(\omega*x))-k*xdot
 (M+m)*\dot{ydot} + m*l*sin(\theta)*\dot{thetadot} = -m*l*cos(theta)*thetadot^2 
        -(M+m)*g- \lambda - k*ydot
 m*l*\cos(\theta)*\dot{xdot}+m*l*\sin(\theta)\dot{\ydot}+m*l^2 * \dot{thetadot} 
        = -m*g*\sin(\theta)
 0 = (-2*x+\omega/3*\sin(\omega*x))*xdot + ydot

where the last constraint comes from deriving the equation of the curve on
which the pendulum slides
"""
from numpy import (arange, zeros, array, sin, cos, asarray, sqrt, pi)
from scikits.odes import dae
import pylab
import os

class Slidingpendulum():
    """ The problem class with the residual function and the constants defined
    """
    
    #default values
    deftend = 120.
    deftstep = 1e-2
    defg = 9.8
    defm = 1.0
    defM = 1.0
    defl = 1.
    defk = 0.5
    defomega = 3.3
    
    #init values
    defx0 = 1.
    defy0 = 1.+cos(defomega)/3.
    deftheta0 = 0.
    defdotx0 = 0.
    defdoty0 = -defg
    defdottheta0 = 0.
    
    def __init__(self, data=None, type='index2'):
        self.tend = Slidingpendulum.deftend
        self.tstep = Slidingpendulum.deftstep
        self.x0 = Slidingpendulum.defx0
        self.y0 = Slidingpendulum.defy0
        self.theta0 = Slidingpendulum.deftheta0
        self.m = Slidingpendulum.defm
        self.M = Slidingpendulum.defM
        self.l = Slidingpendulum.defl
        self.k = Slidingpendulum.defk
        self.g = Slidingpendulum.defg
        self.omega = Slidingpendulum.defomega
        
        self.res = None
        self.jac = None
        
        if data is not None:
            self.tend = data.deftend
            self.tstep = data.deftstep
            self.x0 = data.defx0
            self.y0 = data.defy0
            self.theta0 = data.deftheta0
            self.m = data.defm
            self.M = data.defM
            self.l = data.defl
            self.k = data.defk
            self.g = data.defg
            self.omega = data.defomega
        
        self.stop_t  = arange(.0, self.tend, self.tstep)[1:]
        
        lambdaval = 0.0

        if type == 'index2':
            self.neq = 8
            self.z0 =  array([self.x0, self.y0, self.theta0, 0., 
                              0., 0., lambdaval, lambdaval])
            self.zprime0 = array([0., 0., 0., 0., -self.g, 0., 0., 0.], float)
            self.res = self.resindex2
            self.algvar = array([1,1,1,1,1,1,-1,-1])
            self.exclalg_err = True
        else:
            self.neq = 7
            self.z0 =  array([self.x0, self.y0, self.theta0, 0., 
                              0., 0., lambdaval])
            self.zprime0 = array([0., 0., 0., 0., -self.g, 0., 0.], float)
            self.res = self.resindex1
            self.algvar = array([1,1,1,1,1,1,-1])
            self.exclalg_err = False

    def resindex2(self, tres, yy, yp):
        m = self.m
        M = self.M
        l = self.l
        omega = self.omega
        g = self.g
        k = self.k
        tmp = zeros(self.neq)
        ## x        = yy[0]
        ## y        = yy[1]
        ## theta    = yy[2]
        ## xdot     = yy[3]  = u1
        ## ydot     = yy[4]  = u2
        ## thetadot = yy[5]  = u3
        
        ## it is needed to scale the constraint by blowing up 
        ## the lagrange multiplier with 1e10
        tmp[0]= ((M+m)*yp[3] + m*l*cos(yy[2])*yp[5]  \
                - m*l*sin(yy[2])*yy[5]**2 \
                + yy[6]*1e10*(-2*yy[0]+omega/3.*sin(omega*yy[0]))\
                +k*yy[3])
        tmp[1]= ((M+m)*yp[4] + m*l*sin(yy[2])*yp[5]  \
                + m*l*cos(yy[2])*yy[5]**2 + (M+m)*g \
                + yy[6]*1e10\
                +k*yy[4] )
        tmp[2]= m*l*cos(yy[2])*yp[3] + m*l*sin(yy[2])*yp[4] + m*l**2 * yp[5] \
                + m*g*sin(yy[2])
        #stabalizing factor
        tmp[3]= yp[0] - yy[3] + yy[7]*yy[0]
        tmp[4]= yp[1] - yy[4] + yy[7]*yy[1]
        tmp[5]= yp[2] - yy[5]
        #use both constraint and it's derivative
        tmp[6]= (yy[1]-yy[0]**2-1/3.*cos(omega*yy[0]))/10000.
        tmp[7]= (-2*yy[0] + omega/3.*sin(omega*yy[0]))*yy[3] + yy[4]
        ##print tres, 'yy', yy
        ##print tres, 'yp', yp
        ##print tres, 'res', tmp
        return tmp
    
    def resindex1(self, tres, yy, yp):
        m = self.m
        M = self.M
        l = self.l
        omega = self.omega
        g = self.g
        k = self.k
        tmp = zeros(self.neq)
        ## x        = yy[0]
        ## y        = yy[1]
        ## theta    = yy[2]
        ## xdot     = yy[3]
        ## ydot     = yy[4]
        ## thetadot = yy[5]
        
        ## it is needed to scale the constraint by blowing up 
        ## the lagrange multiplier with 1e10
        tmp[0]= (M+m)*yp[3] + m*l*cos(yy[2])*yp[5]  \
                - m*l*sin(yy[2])*yy[5]**2 \
                + yy[6]*1e10*(-2*yy[0]+omega/3.*sin(omega*yy[0]))\
                +k*yy[3]
        tmp[1]= (M+m)*yp[4] + m*l*sin(yy[2])*yp[5]  \
                + m*l*cos(yy[2])*yy[5]**2 + (M+m)*g \
                + yy[6]*1e10\
                +k*yy[4] 
        tmp[2]= m*l*cos(yy[2])*yp[3] + m*l*sin(yy[2])*yp[4] + m*l**2 * yp[5] \
                + m*g*sin(yy[2])
        tmp[3]= yp[0] - yy[3]
        tmp[4]= yp[1] - yy[4]
        tmp[5]= yp[2] - yy[5]
        #following equation is wrong, we need to derive another time and 
        #eliminate the dot{xdot} and dot{ydot} so that lagrange multiplier appears
        tmp[6]= (-2*yy[0] + omega/3.*sin(omega*yy[0]))*yy[3] + yy[4]
        ##sys.exit()
        ##print tres, 'yy', yy
        ##print tres, 'yp', yp
        ##print tres, 'res', tmp
        return tmp

def main():
    """
    The main program: instantiate a problem, then use odes package to solve it
    """
    input = raw_input("Solve as\n 1 = index 2 problem\n 2 = index 1 problem\n"
                " \n 4 = info\n\n"
                "Answer (1,2 or 4) : ")
    if input == '1':
        problem = Slidingpendulum(type='index2')
    elif input == '2':
        problem = Slidingpendulum(type='index1')
    else:
        print __doc__
        return
    
    input1 = raw_input("Solve with\n 1 = ddaspk\n 2 = ida\n\n"
                       "Answer (1 or 2) : ").strip()
    if input1 not in ["1", "2"]:
        print "Invalid solution method given"
        return
    
    z = [0]*(1+len(problem.stop_t)); zprime = [0]*(1+len(problem.stop_t))

    ig = dae(problem.res, problem.jac)
    #first compute the correct initial condition from the values of z0
    if input1 == "2":
        ig.set_integrator('odesIDA',algebraic_var=problem.algvar,
                        compute_initcond='yode0',
                        first_step=1e-9,
                        atol=1e-5,rtol=1e-5)
    if input1 == "1":
        #first compute the correct initial condition from the values of z0
        ig.set_integrator('ddaspk',algebraic_var=problem.algvar,
                    compute_initcond='yode0',
                    first_step=1e-9,
                    atol=1e-2,rtol=1e-2)
    ig.set_initial_value(problem.z0, problem.zprime0,  t=0.0)

    i=0
    z[i],  zprime[i] = ig.solve(1e-9);
    assert ig.successful(), (problem,)
    print 'started from z0 = ', problem.z0
    print 'initial condition calculated, [z,zprime] = [', z[0], zprime[0], ']'

    if input1 == "2":
        ig.set_integrator('odesIDA',algebraic_var=problem.algvar,
                        first_step=1e-9,
                        atol=1e-5,
                        rtol=1e-5,
                        exclude_algvar_from_error=problem.exclalg_err,
                        nsteps = 1500)
    if input1 == "1":
        ig.set_integrator('ddaspk',algebraic_var=problem.algvar,
                        first_step=1e-9,
                        atol=1e-2,
                        rtol=1e-2,
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
                print 'Error in solver, breaking solution at time %g' % time
                break


    print 'last sol', z[i-1], zprime[i-1]
    print 'has residual: ', problem.res(problem.stop_t[i-2], z[i-1], 
                                        zprime[i-1])

    nr = i
    xt = [z[i][0] for i in range(nr)]
    yt = [z[i][1] for i in range(nr)]
    thetat = [z[i][2] for i in range(nr)]
    time = zeros(nr,float)
    time[0] = 0.0
    if error:
        time[1:]  = problem.stop_t[:nr-1] 
    else:
        time[1:]  = problem.stop_t[:nr]
        
    pylab.figure(1)
    pylab.subplot(111)
    pylab.scatter(xt, yt)
    pylab.scatter(xt + problem.l*sin(thetat), yt -  problem.l*cos(thetat))
    pylab.axis('equal')
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
        x2 = sol[0]+problem.l*sin(sol[2])
        y2 = sol[1]-problem.l*cos(sol[2])
        line(ax, sol[0], sol[1], x2, y2)
        circle(ax, x2, y2, 0.1)

    def drawonesol(nr, sizex, sizey, ext=None):
        pylab.clf()
        a = pylab.axes()
        draw(a, nr)
        pylab.axis('scaled')
        pylab.xlim(-sizex, sizex)
        pylab.ylim(-sizey, sizey)
        if ext is None:
            ext = nr
        pylab.savefig('figsslidingpendulum' + os.sep + 'outsol%08i.png' % ext)

    def open_file_with_default_application( file_path ):
        """
        Launch a program to open an arbitrary file. The file will be opened using 
        whatever program is configured on the host as the default program for that 
        type of file.
        """
        
        norm_path = os.path.normpath( file_path )
        
        if not os.path.exists(norm_path):
            print "%s does not exist" % file_path
            return
            
        if os.sys.platform == 'win32':
            try:
                os.startfile(norm_path)
            except WindowsError, msg:
                print "Error Opening File. " + str(msg)
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
        if os.path.isdir('figsslidingpendulum'):
            shutil.rmtree('figsslidingpendulum')
        os.mkdir('figsslidingpendulum')
        if not os.path.isdir('anislidingpendulum'):
            os.mkdir('anislidingpendulum')
        
        pylab.figure(2)
        secs = 0
        frame = 0
        print 'Generating output ...\n'
        for solnr in range(0,nr,5):
            drawonesol(solnr, sizex, sizey, frame)
            frame += 1
            if solnr // 500 <> secs :
                secs = solnr // 500
                print '     ... at %i seconds ' % (secs * 5 ) 

        print 'Creating movie using ffmpeg with output ... \n'
        import subprocess
        subprocess.call(['ffmpeg', '-r', '20', '-i', 'figsslidingpendulum' + os.sep + 
                        'outsol%8d.png',  '-f',  'avi', '-y', 
                        'anislidingpendulum' + os.sep + 
                                            'slidingpendulum'+ext+'.mpg'])
        #remove unused pictures
        shutil.rmtree('figsslidingpendulum')
        #opening movie with default player
        print 'Opening user with default application ... \n'
        open_file_with_default_application('anislidingpendulum' + os.sep +
                    'slidingpendulum'+ext+'.mpg')

    input2 = raw_input('Create animation of the solution? (y/n): ')
    print('\n')
    if (input2 == 'y' or input2 == 'yes'):
        create_animation(1.+problem.l, 1+problem.l, '0')

if __name__ == "__main__":
    main()
