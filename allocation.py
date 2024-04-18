from gurobipy import *

# List of organs
omega = 3

# Organ arrival times
organ_times = [1, 2, 3]

# List of patients
n = 3

# Patient arrival times
E_times = [1, 1, 1]

# Patient death times
M_times = [2, 2, 3]

# Compute value matrix based on arrival and death times
dict = {}
for o in range(omega):
    for p in range(n):
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
organ_cstr = m.addConstrs((x.sum() == 1 for j in range(n)), name='organ')

# Patient constraints

