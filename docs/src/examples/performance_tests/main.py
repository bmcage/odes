from __future__ import print_function
import time
from collections import OrderedDict, namedtuple
import numpy as np
from pandas import DataFrame
from scipy.integrate import odeint, ode
#new
HAS_SOLVEIVP = False
try:
    from scipy.integrate import solve_ivp
    HAS_SOLVEIVP = True
except:
    pass
if not HAS_SOLVEIVP:
    try:
        from scipy_ode import solve_ivp
        HAS_SOLVEIVP = True
    except:
        pass
if not HAS_SOLVEIVP:
    print ("\nWARNING: solve_ivp not available, scipping those methods!\n")

HAS_ODES = False
try:
    from scikits.odes.odeint import odeint as odes_odeint
    from scikits.odes import ode as odes_ode
    HAS_ODES = True
except:
    pass
if not HAS_ODES:
    print ("\nWARNING: scikits.odes not available, scipping cvode methods!\n")

import ggplot as gg
import egfngf_model

models = [
    egfngf_model
]

class scipy_ode_int:
    name = 'odeint'

    def __call__(self, model, rtol):
        def reordered_ode(t, y):
            return model.f(y, t, model.k)
        result = odeint(reordered_ode, model.y0, model.ts, rtol=rtol)
        return result

class scipy_ode_class:
    def __init__(self, name):
        self.name = name

        space_pos = name.find(" ")
        if space_pos > -1:
            self.solver = name[0:space_pos]
            self.method = name[space_pos+1:]
        else:
            self.solver = name
            self.method = None

    def __call__(self, model, rtol):
        solver = ode(model.f)
        solver.set_integrator(self.solver, method=self.method, rtol=rtol,
                              nsteps=10000)
        solver.set_initial_value(model.y0, 0.0)
        solver.set_f_params(model.k)

        result = np.empty((len(model.ts), len(model.y0)))
        for i, t in enumerate(model.ts):  # Drop t=0.0
            if t == 0:
                result[i, :] = model.y0
                continue
            result[i, :] = solver.integrate(t)

        return result

class scipy_odes_class(scipy_ode_class):

    def __call__(self, model, rtol):
        solver = odes_ode(self.solver, model.f_odes, old_api=False,
                          lmm_type=self.method, rtol=rtol,
                          user_data = model.k)
        solution = solver.solve(model.ts, model.y0)
        for i, t in enumerate(model.ts):
            try:
                result[i, :] = solution.values.y[i]
            except:
                # no valid solution anymore
                result[i, :] = 0

        return result

class scipy_solver_class:
    def __init__(self, name):
        self.name = name

    def __call__(self, model, rtol):
        def collected_ode(t, y):
            return model.f(t, y, model.k)

        sol = solve_ivp(collected_ode, [0.0, np.max(model.ts)], model.y0, method=self.name, rtol=rtol, t_eval=model.ts)

        return sol.y.transpose()

methods = [
    scipy_ode_int(),
    scipy_ode_class("vode bdf"),
    scipy_ode_class("vode adams"),
    scipy_ode_class("lsoda"),
    scipy_ode_class("dopri5"),
    scipy_ode_class("dop853"),
    ]
if HAS_SOLVEIVP:
    methods += [scipy_solver_class("RK45"),
                scipy_solver_class("RK23"),
                scipy_solver_class("Radau"),
                scipy_solver_class("BDF"),
                ]
if HAS_ODES:
    methods += [scipy_odes_class("cvode BDF"),
                scipy_odes_class("cvode ADAMS"),
               ]

rtols = 10 ** np.arange(-9.0, 0.0)

GoldStandard = namedtuple('GoldStandard', ['name', 'values', 'max'])

gold_standards = []
for model in models:
    print('Gold standard for {}'.format(model.name))
    result = methods[0](model, 1e-12)
    gold_standards.append((model.name, GoldStandard(model.name, result, np.max(result))))

gold_standards = OrderedDict(gold_standards)

data = []
for method in methods:
    for model in models:
        for rtol in rtols:
            print('method: {} model: {} rtol: {}'.format(method.name, model.name, rtol), end='')

            # Run
            tic = time.time()
            result = method(model, rtol)
            toc = time.time() - tic

            # Compare to gold standard
            standard = gold_standards[model.name]
            diff = result - standard.values
            max_rel_diff = np.max(diff/standard.max)

            # Append to table
            record = (method.name, model.name, rtol, max_rel_diff, toc)
            print(' err: {} toc: {}'.format(max_rel_diff, toc))
            data.append(record)


data = DataFrame(data, columns=['method', 'model', 'rtol', 'err', 'time'])

print(gg.ggplot(data, gg.aes(x='err', y='time', color='method'))
      + gg.geom_point(size=60.0)
      + gg.geom_line()
      + gg.scale_x_log()
      + gg.scale_y_log()
      + gg.xlim(1e-10, 1e-2))
