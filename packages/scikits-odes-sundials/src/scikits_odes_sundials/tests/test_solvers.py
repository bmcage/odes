import numpy as np
import scikits_odes_sundials as sun


def ode(t, y, yp):
    yp[0] = 0.1
    yp[1] = y[1]


def dae(t, y, yp, res):
    res[0] = yp[0] - 0.1
    res[1] = 2*y[0] - y[1]


def ode_solution(t, y0):
    y = np.zeros([t.size, 2])
    y[:, 0] = 0.1*t + y0[0]
    y[:, 1] = y0[1]*np.exp(t)
    return y


def dae_solution(t, y0):
    y = np.zeros([t.size, 2])
    y[:, 0] = 0.1*t + y0[0]
    y[:, 1] = 2*y[:, 0]
    return y


# common times and initial condition
times = np.array([0, 5, 10])
y0 = np.array([1, 2])


# Testing cvode for ode
solver = sun.cvode.CVODE(ode, rtol=1e-9, atol=1e-12)
solution = solver.solve(times, y0)

print("\nCVODE", "-"*10, solution, sep="\n")
assert np.allclose(solution.values.y, ode_solution(times, y0))


# Testing cvodes for ode
solver = sun.cvodes.CVODES(ode, rtol=1e-9, atol=1e-12)
solution = solver.solve(times, y0)

print("\nCVODES", "-"*10, solution, sep="\n")
assert np.allclose(solution.values.y, ode_solution(times, y0))


# Testing ida for dae
solver = sun.ida.IDA(dae, algebraic_vars_idx=[1], compute_initcond='yp0')
solution = solver.solve(times, y0, [0, 0])

print("\nIDA", "-"*10, solution, sep="\n")
assert np.allclose(solution.values.y, dae_solution(times, y0))


# Testing idas for dae
solver = sun.idas.IDAS(dae, algebraic_vars_idx=[1], compute_initcond='yp0')
solution = solver.solve(times, y0, [0, 0])

print("\nIDAS", "-"*10, solution, sep="\n")
assert np.allclose(solution.values.y, dae_solution(times, y0))
