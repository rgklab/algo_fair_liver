import gurobipy as gp
from gurobipy import GRB, LinExpr
import numpy as np

def discretize_diff(diff):
    if diff <= 3: return 1  
    if (diff > 3) & (diff <= 7): return 0.9
    if diff > 7: return 0.75

def allocate_organs(
    organs, patients, organ_times, E_times, M_times, 
    obj, flag_liver_size=False, alpha=0.5,
    women_indices=None, small_organ_indices=None,
    flag_print_allocation=False, flag_discrete_t=False
):
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
        for _, o in enumerate(organs):
            for _, p in enumerate(patients):
                assignment_time = organ_times[o]
                if E_times[p] <= assignment_time <= M_times[p]:
                    if flag_discrete_t: 
                        val = discretize_diff(M_times[p]-assignment_time)
                    else:
                        val = 1
                    if p in women_indices:
                        if o in small_organ_indices: 
                            value_dict[(o, p)] = val + alpha*1
                        else:
                            value_dict[(o, p)] = val
                    else:
                        if o in small_organ_indices: 
                            value_dict[(o, p)] = val + alpha*0.5
                        else:
                            value_dict[(o, p)] = val + alpha*1
                else:
                    value_dict[(o, p)] = 0

    assignment, value = gp.multidict(value_dict)

    # Compute envy matrix if necessary
    if obj == 'envy':
        envy = {}
        for p in patients:
            for o in organs:
                envy_op = 0
                value_o = value[(o, p)]
                for q in organs:
                    if q != o:
                        value_q = value[(q, p)]
                        if value_q > value_o:
                            envy_op += value_q - value_o
                envy[(o, p)] = envy_op

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

    # Objective function
    if obj == 'welfare':
        m.setObjective(x.prod(value), GRB.MAXIMIZE)
    elif obj == 'envy':
        m.setObjective(x.prod(value) - (1/n) * x.prod(envy), GRB.MAXIMIZE)
    else:
        raise ValueError(f"{obj} is not a valid objective function!")

    m.optimize()
    n_organ_allocate = 0
    if flag_print_allocation:
        if m.status == GRB.OPTIMAL:
            print("Optimal solution found:")
            for i in range(n):
                for j in range(omega):
                    if x[j+1, i+1].x > 0.5:  # Print only allocations with value > 0.5
                        print(f"Agent {i} gets item {j}")
                        n_organ_allocate += 1
            print(f'num organs allocation: {n_organ_allocate}')
        else:
            print("No optimal solution found")

    return m, x, value_dict

def update_model(m, organs, patients, new_organs, new_patients, allocations):
    # Remove variables for allocated organs and patients
    for (o, p) in allocations.keys():
        m.remove(m.getVarByName(f'x[{o},{p}]'))

    # Add new variables for new organs and patients
    for o in new_organs:
        for p in patients + new_patients:
            m.addVar(vtype=GRB.BINARY, name=f'x[{o},{p}]')
    
    for p in new_patients:
        for o in organs + new_organs:
            m.addVar(vtype=GRB.BINARY, name=f'x[{o},{p}]')

    m.update()
    #m.optimize()
    return m

def gen_data(T, omega, n, init_t=0):
    organ_t_rand = np.random.choice(T, omega, replace=False) + 1 + init_t
    organ_t_rand = np.sort(organ_t_rand)
    E_times_rand = np.random.randint(1, T, n)
    E_times_rand = np.sort(E_times_rand)
    women_indices = np.random.choice(n, int(0.4 * n), replace=False) + 1
    small_organ_indices = np.random.choice(omega, int(0.25 * n), replace=False) + 1
    return organ_t_rand, E_times_rand, women_indices, small_organ_indices

   
if __name__ == '__main__':
    # List of organs
    T = 100
    init_T = 20 #100 time points
    add_T = 10
    init_omega = 8
    add_omega = 3
    init_n = 12
    add_n = 5
    flag_organ_size = True
    organ_t_rand, E_times_rand, women_indices, small_organ_indices = gen_data(init_T, init_omega, init_n)
    patients = list(np.arange(1, init_n + 1, 1))
    organs = list(np.arange(1, init_omega + 1, 1))
    E_times = {}
    M_times = {}
    organ_times = {}
    for i, Ei in enumerate(E_times_rand):
        E_times[i+1] = Ei
        Mi = np.random.randint(Ei, T, 1)
        M_times[i+1] = Mi[0]
    for i, oi in enumerate(organ_t_rand):
        organ_times[i+1] = oi

    m, x, value_dict = allocate_organs(
        organs, patients, organ_times, E_times, M_times, 
        obj='envy', flag_liver_size=True, 
        women_indices=women_indices, small_organ_indices=small_organ_indices,
        flag_print_allocation=False, flag_discrete_t=True
    )
    n_range = add_T
    available_patients = patients
    available_organs = organs
    for j in range((T-init_T)//add_T):
        organ_range = organ_t_rand <= n_range
        n_organ_all = 0 
        new_organs, new_patients = [], []
        # Remove allocated organs and patients
        for (o, p) in value_dict:
            if x[o, p].X > 0.5 and p in available_patients and o in available_organs:
                available_patients.remove(p)
                available_organs.remove(o)

        for i in patients:
            for j in range(sum(organ_range)):
                if x[j+1, i+1].x > 0.5:
                    print(f"Agent {i} gets item {j}")
                    n_organ_all += 1
                    ### ToDO: find previously alloacted organs and people in add_T time and remove them from the system
                    ### remove them from value_dict
        print(f'num organs allocated until time {n_range}')
        new_organ_t, new_E_times, new_w_indices, new_s_organ_indices = gen_data(add_T, add_omega, add_n, init_t=init_T)
        new_patients.extend(range(max(patients)+1, max(patients)+1+add_n))
        new_organs.extend(range(max(organs)+1, max(organs)+1+add_omega))
        organ_times.update({o: t for o, t in zip(new_organs, new_organ_t)})
        E_times.update({p: t for p, t in zip(new_patients, new_E_times)})
        M_times.update({p: np.random.randint(E_times[p], T) for p in new_patients})

        m = update_model(m, organs, patients, new_organs, new_patients, allocations)
        available_patients.update(new_patients)
        available_organs.update(new_organs)
        patients.extend(new_patients)
        organs.extend(new_organs)
        init_T += add_T
        n_range += add_T


