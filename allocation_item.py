import gurobipy as gp
from gurobipy import GRB, LinExpr
import numpy as np

def allocate_organs(organs, patients, organ_times, E_times, M_times, obj, flag_liver_size=False, women_indices=None, small_organ_indices=None):
    """
    @param obj: objective to maximize
        welfare: only consider sum of all values
        envy: welfare penalized by envy between patients
    """
    # Compute value matrix based on arrival and death times
    omega = len(organs)
    n = len(patients)

    value_dict = {}
    for _, o in enumerate(organs):
        for _, p in enumerate(patients):
            assignment_time = organ_times[o]
            if E_times[p] <= assignment_time <= M_times[p]:
                value_dict[(o, p)] = 1
            else:
                value_dict[(o, p)] = 0
                
    if flag_liver_size:
        alpha = 0.5
        for _, o in enumerate(organs):
            for _, p in enumerate(patients):
                assignment_time = organ_times[o]
                if E_times[p] <= assignment_time <= M_times[p]:
                    if p in women_indices:
                        if o in small_organ_indices: 
                            value_dict[(o, p)] = 1 + alpha*1
                        else:
                            value_dict[(o, p)] = 1
                    else:
                        if o in small_organ_indices: 
                            value_dict[(o, p)] = 1 + alpha*0.5
                        else:
                            value_dict[(o, p)] = 1 + alpha*1
                else:
                    value_dict[(o, p)] = 0

    assignment, value = gp.multidict(value_dict)

    # Declare and initialize model
    m = gp.Model('organ_alloc')

    # Decision variables
    x = m.addVars(assignment, name='assign')

    # Add envy constraints
    """
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip comparing an agent to themselves
                # Agent i should not prefer the allocation of agent j
                envy_free_constraint = (gp.quicksum(x[k+1, i+1] * value[k+1, i+1] for k in range(omega)) >=
                                        gp.quicksum(x[k+1, j+1] * value[k+1, i+1] for k in range(omega)))
                
                m.addConstr(envy_free_constraint, name=f"envy_free_{i}_{j}")
    """
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

    return m, x


if __name__ == '__main__':
    # List of organs
    T = 100 #100 time points
    omega = 30 #num organs
    n = 50 #num agents
    flag_organ_size = True
    #organ_times = np.random.randint(1, T, n_organs)
    organ_t_rand = np.random.choice(T, omega, replace=False) + 1
    organ_t_rand = np.sort(organ_t_rand)
    E_times_rand = np.random.randint(1, T, n)
    E_times_rand = np.sort(E_times_rand)

    patients = list(np.arange(1, n + 1, 1))
    organs = list(np.arange(1, omega + 1, 1))
    women_indices = np.random.choice(n, int(0.4 * n), replace=False) + 1
    small_organ_indices = np.random.choice(omega, int(0.25 * n), replace=False) + 1

    E_times = {}
    M_times = {}
    organ_times = {}
    for i, Ei in enumerate(E_times_rand):
        E_times[i+1] = Ei
        Mi = np.random.randint(Ei, T, 1)
        M_times[i+1] = Mi[0]
    for i, oi in enumerate(organ_t_rand):
        organ_times[i+1] = oi

   # m = allocate_organs(organs, patients, organ_times, E_times, M_times, obj='welfare')
    m, x = allocate_organs(organs, patients, organ_times, E_times, M_times, obj='welfare', flag_liver_size=True, women_indices=women_indices, small_organ_indices=small_organ_indices)

    flag_print_allocation = True
    n_organ_allocate = 0 
    if flag_print_allocation:
        if m.status == GRB.OPTIMAL:
            print("Optimal solution found:")
            for i in range(n):
                for j in range(omega):
                    if x[j+1, i+1].x > 0.5:  # Print only allocations with value > 0.5
                        print(f"Agent {i} gets item {j}")
                        n_organ_allocate += 1

        else:
            print("No optimal solution found")
        print(n_organ_allocate)
