from gurobipy import *

# List of organs
omega = 3
organs = [o for o in range(1, omega + 1)]

# Organ arrival times
organ_times = {
    1: 1,
    2: 2,
    3: 3
}

# List of patients
n = 3
patients = [p for p in range(1, n + 1)]

# Patient arrival times
E_times = {
    1: 1,
    2: 1,
    3: 1
}

# Patient death times
M_times = {
    1: 2,
    2: 2,
    3: 3
}

# Compute value matrix based on arrival and death times
dict = {}
for o, _ in enumerate(organs):
    for p, _ in enumerate(patients):
        assignment_time = organ_times[o]
        if E_times[p] <= assignment_time <= M_times[p]:
            dict{(o, p)} = 1
        else:
            dict{(o, p)} = 0

assignment, value = multidict(dict)

# Declare and initialize model
m = Model('organ_alloc')

# Decision variables
x = m.addVars(assignment, name='assign')

# Organ constraints
# Each organ is assigned to exactly one patient
organ_cstr = m.addConstrs((x.sum(o, '*') == 1 for o in organs), name='organ')

# Patient constraints
# Each patient is assigned at most one organ
patient_cstr = m.addConstrs((x.sum('*', p) <= 1 for p in patients), name='patient')

# Objective function
m.setObjective(x.prod(value), GRB.MAXIMIZE)

m.optimize()
