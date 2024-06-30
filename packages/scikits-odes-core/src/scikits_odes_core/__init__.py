from . import _version
__version__ = _version.get_versions()['version']


class DaeBase:
    """
    The interface which DAE solvers must implement.

    Parameters
    ----------
    Rfn :
        residual function
    options : mapping
        Additional options for initialization, solver dependent
    """

    integrator_classes = []

    def __init__(self, Rfn, **options):
        raise NotImplementedError('all DAE solvers must implement this')

    def set_options(self, **options):
        """
        Set specific options for the solver.

        Calling set_options a second time, normally resets the solver.
        """
        raise NotImplementedError('all DAE solvers must implement this')

    def solve(self, tspan, y0,  yp0):
        """
        Runs the solver.

        Parameters
        ----------
            tspan : list/array
                A list of times at which the computed value will be returned.
                Must contain the start time as first entry.
            y0 : list/array
                List of initial values
            yp0 : list/array
                List of initial values of derivatives

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnumXXX)
            ``values``  Named tuple with entries array t and array y and array ydot. y will correspond to y_retn value and ydot to yp_retn!
            ``errors``  Named tuple with entries t and y and ydot of error
            ``roots``   Named tuple with entries array t and array y and array ydot
            ``tstop``   Named tuple with entries array t and array y and array ydot
            ``message`` String with message in case of an error
            =========== ==========================================

        old_api is True : tuple
            tuple with the following elements in order

            ========== ==========================================
            Field      Meaning
            ========== ==========================================
            ``flag``   indicating return status of the solver
            ``t``      numpy array of times at which the computations were successful
            ``y``      numpy array of values corresponding to times t (values of y[i, :] ~ t[i])
            ``yp``     numpy array of derivatives corresponding to times t (values of yp[i, :] ~ t[i])
            ``t_err``  float or None - if recoverable error occurred (for example reached maximum number of allowed iterations), this is the time at which it happened
            ``y_err``  numpy array of values corresponding to time t_err
            ``yp_err`` numpy array of derivatives corresponding to time t_err
            ========== ==========================================

        """
        raise NotImplementedError('all DAE solvers must implement this')

    def init_step(self, t0, y0, yp0, y_ic0_retn = None, yp_ic0_retn = None):
        """
        Initializes the solver and allocates memory.

        Parameters
        ----------
        t0 : number
            initial time
        y0 : list/array
            initial condition for y
        yp0 :  list/array
            initial condition for yp
        y_ic0 : numpy array
            (optional) returns the calculated consistent initial condition for y
        yp_ic0 :  numpy array
            (optional) returns the calculated consistent initial condition for y
            derivated.

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnumXXX)
            ``values``  Named tuple with entries t and y and ydot. y will correspond to y_retn value and ydot to yp_retn!
            ``errors``  Named tuple with entries t and y and ydot
            ``roots``   Named tuple with entries t and y and ydot
            ``tstop``   Named tuple with entries t and y and ydot
            ``message`` String with message in case of an error
            =========== ==========================================

        old_api is True : tuple
            tuple with the following elements in order

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    status of the computation (successful or error occurred)
            ``t_out``   time, where the solver stopped (when no error occurred, t_out == t)
            =========== ==========================================

        """
        raise NotImplementedError('all DAE solvers must implement this')

    def step(self, t, y_retn=None, yp_retn=None):
        """
        Method for calling successive next step of the IDA solver to allow
        more precise control over the IDA solver. The 'init_step' method has to
        be called before the 'step' method.

        A step is done towards time t, and output at t returned. This time can
        be higher or lower than the previous time. If option
        'one_step_compute'==True, and the solver supports it, only one internal
        solver step is done in the direction of t starting at the current step.

        If old_api=True, the old behavior is used: if t>0.0 then integration is
        performed until this time and results at this time are returned in
        y_retn; else if if t<0.0 only one internal step is performed towards
        time abs(t) and results after this one time step are returned.

        Parameters
        ----------
        t : number
        y_retn : numpy array (ndim = 1) or None.
            (Needs to be preallocated) If not None, will be filled with y at
            time t. If None y_retn is not used.
        yp_retn : numpy array (ndim = 1) or None.
            (Needs to be preallocated) If not None, will be filled with
            derivatives of y at time t. If None yp_retn is not used.

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnumXXX)
            ``values``  Named tuple with entries t and y and ydot. y will correspond to y_retn value and ydot to yp_retn!
            ``errors``  Named tuple with entries t and y and ydot
            ``roots``   Named tuple with entries t and y and ydot
            ``tstop``   Named tuple with entries t and y and ydot
            ``message`` String with message in case of an error
            =========== ==========================================

        old_api is True : tuple
            tuple with the following elements in order

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    status of the computation (successful or error occurred)
            ``t_out``   time, where the solver stopped (when no error occurred, t_out == t)
            =========== ==========================================

        """
        raise NotImplementedError('all DAE solvers must implement this')


class OdeBase:
    """
    The interface which ODE solvers must implement.

    Parameters
    ----------
    Rfn : function
        A function which computes the required derivatives. The signature
        should be ``func(t, y, y_dot, *args, **kwargs)``. Note that *args and
        **kwargs handling are solver dependent.

    options : mapping
        Additional options for initialization, solver dependent
    """

    integrator_classes = []

    def __init__(self, Rfn, **options):
        raise NotImplementedError('all ODE solvers must implement this')

    def set_options(self, **options):
        """
        Set specific options for the solver.

        Calling set_options a second time, normally resets the solver.
        """
        raise NotImplementedError('all ODE solvers must implement this')

    def solve(self, tspan, y0):
        """
        Runs the solver.

        Parameters
        ----------
        tspan : array (or similar)
            a list of times at which the computed value will be returned. Must
            contain the start time as first entry.

        y0 : array (or similar)
            a list of initial values

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnum)
            ``values``  Named tuple with entries t and y
            ``errors``  Named tuple with entries t and y
            ``roots``   Named tuple with entries t and y
            ``tstop``   Named tuple with entries t and y
            ``message`` String with message in case of an error
            =========== ==========================================

        old_api is True : tuple
            tuple with the following elements in order

            ========== ==========================================
            Field      Meaning
            ========== ==========================================
            ``flag``   indicating return status of the solver
            ``t``      numpy array of times at which the computations were successful
            ``y``      numpy array of values corresponding to times t (values of y[i, :] ~ t[i])
            ``t_err``  float or None - if recoverable error occurred (for example reached maximum number of allowed iterations), this is the time at which it happened
            ``y_err``  numpy array of values corresponding to time t_err
            ========== ==========================================
        """
        raise NotImplementedError('all ODE solvers must implement this')

    def init_step(self, t0, y0):
        """
        Initializes the solver and allocates memory.

        Parameters
        ----------
        t0 : number
            initial time
        y0 : array
            initial condition for y (can be list or numpy array)

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnum)
            ``values``  Named tuple with entries t and y
            ``errors``  Named tuple with entries t and y
            ``roots``   Named tuple with entries t and y
            ``tstop``   Named tuple with entries t and y
            ``message`` String with message in case of an error
            =========== ==========================================

        old_api is True : tuple
            tuple with the following elements in order

            ========== ==========================================
            Field      Meaning
            ========== ==========================================
            ``flag``   boolean status of the computation (successful or error occurred)
            ``t_out``  initial time
            ========== ==========================================
        """
        raise NotImplementedError('all ODE solvers must implement this')

    def step(self, t, y_retn=None):
        """
        Method for calling successive next step of the ODE solver to allow
        more precise control over the solver. The 'init_step' method has to
        be called before the 'step' method.

        A step is done towards time t, and output at t returned.  This time can
        be higher or lower than the previous time.  If option
        'one_step_compute'==True, and the solver supports it, only one internal
        solver step is done in the direction of t starting at the current step.

        If old_api=True, the old behavior is used: if t>0.0 then integration is
        performed until this time and results at this time are returned in
        y_retn if t<0.0 only one internal step is performed towards time abs(t)
        and results after this one time step are returned

        Parameters
        ----------
        t : number

        y_retn : numpy vector (ndim = 1)
            in which the computed value will be stored  (needs to be
            preallocated).  If None y_retn is not used.

        Returns
        -------
        old_api is False : namedtuple
            namedtuple with the following attributes

            =========== ==========================================
            Field       Meaning
            =========== ==========================================
            ``flag``    An integer flag (StatusEnum)
            ``values``  Named tuple with entries t and y
            ``errors``  Named tuple with entries t and y
            ``roots``   Named tuple with entries t and y
            ``tstop``   Named tuple with entries t and y
            ``message`` String with message in case of an error
            =========== ==========================================

        old_api is True : tuple
            tuple with the following elements in order

            ========== ==========================================
            Field      Meaning
            ========== ==========================================
            ``flag``   status of the computation (successful or error occurred)
            ``t_out``  time, where the solver stopped (when no error occurred, t_out == t)
            ========== ==========================================
        """
        raise NotImplementedError('all ODE solvers must implement this')
