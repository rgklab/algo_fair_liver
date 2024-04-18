"""
Allocating to three job positions: tester, developer, and architect to 
three resources: Carlos, Joe, and Monika.

Each resource has a different affinity score for each job position. 
The objective is to maximize the overall affinity score of the assignments.
"""

from gurobipy import *

# Set up data
# List of resources
R = ['Carlos', 'Joe', 'Monika']
# List of jobs
J = ['Tester', 'Developer', 'Architect']

# Value matrix that encodes affinity scores
combinations, score = multidict({
    ('Carlos', 'Tester'): 53,
    ('Carlos', 'Developer'): 27,
    ('Carlos', 'Architect'): 13,
    ('Joe', 'Tester'): 80,
    ('Joe', 'Developer'): 47,
    ('Joe', 'Architect'): 67,
    ('Monika', 'Tester'): 53,
    ('Monika', 'Developer'): 73,
    ('Monika', 'Architect'): 47
})

# Declare and initialize model
m = Model('RAP')

# Decision variables
# x = m.addVars(combinations, vtype=GRB.BINARY, name='assign')
# Not sure if the GRB.BINARY is necessary here since this is a perfect problem?
x = m.addVars(combinations, name='assign')

# Constraints
# Job constraint: Each job is assigned exactly one resource.
job_cstr = m.addConstrs((x.sum('*', j) == 1 for j in J), name='job')

# Resource constraint: Each resource is assigned to at most one job.
resource_cstr = m.addConstrs((x.sum(r, '*') <= 1 for r in R), name='resource')

# Objective function
m.setObjective(x.prod(score), GRB.MAXIMIZE)

m.optimize()

