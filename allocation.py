import gurobipy as gp
from gurobipy import GRB, LinExpr


def allocate_organs(organs, patients, organ_times, E_times, M_times, envy_multiplier, obj):
    """
    @param obj: objective to maximize
        welfare: only consider sum of all values
        envy: welfare penalized by envy between patients
    """
    # Compute value matrix based on arrival and death times
    value_dict = {}
    for _, o in enumerate(organs):
        for _, p in enumerate(patients):
            assignment_time = organ_times[o]
            if E_times[p] <= assignment_time <= M_times[p]:
                value_dict[(o, p)] = 1
            else:
                value_dict[(o, p)] = 0

    assignment, value = gp.multidict(value_dict)

    # Declare and initialize model
    m = gp.Model('organ_alloc')

    # Decision variables
    x = m.addVars(assignment, name='assign')

    # Organ constraints
    # Each organ is assigned to exactly one patient
    organ_cstr = m.addConstrs((x.sum(o, '*') == 1 for o in organs), name='organ')

    # Patient constraints
    # Each patient is assigned at most one organ
    patient_cstr = m.addConstrs((x.sum('*', p) <= 1 for p in patients), name='patient')

    def envy(x, value):
        total_envy = LinExpr()
        # import pdb; pdb.set_trace()
        # for p in patients:
        #     my_value_my_organ = sum(x.prod(value)[:, p])
        #     for o in organs:
        #         my_value_other_organ = value_dict[(o, p)]
        #         if my_value_other_organ > my_value_my_organ:
        #             total_envy += my_value_other_organ - my_value_my_organ
        return total_envy

    # Objective function
    if obj == 'welfare':
        m.setObjective(x.prod(value), GRB.MAXIMIZE)
    elif obj == 'envy':
        m.setObjective(x.prod(value) - envy_multiplier * envy(x, value), GRB.MAXIMIZE)
    else:
        raise ValueError(f"{obj} is not a valid objective function!")

    m.optimize()


if __name__ == '__main__':
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

    allocate_organs(organs, patients, organ_times, E_times, M_times, obj='envy')