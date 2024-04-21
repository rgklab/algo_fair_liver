import gurobipy as gp
from gurobipy import GRB, LinExpr
import numpy as np

def discretize_diff(diff):
    if diff <= 3: return 1  
    if (diff > 3) & (diff <= 7): return 0.9
    if diff > 7: return 0.75

def create_value_dict(organs, patients, organ_times, E_times, M_times, flag_liver_size=False, 
                      alpha=0.5, women_indices=None, small_organ_indices=None, flag_discrete_t=False):
    # Compute value matrix based on arrival and death times
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
    return value_dict

def allocate_organs(organs, patients, obj, value_dict, flag_print_allocation=False):
    """
    @param obj: objective to maximize
        welfare: only consider sum of all values
        envy: welfare penalized by envy between patients
    """
    omega = len(organs)
    n = len(patients)
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

    return m, x

def update_model(m, x, organs, patients, new_organs, new_patients, value_dict, rm_list, value_dict_new, obj='envy'):
    # Remove variables for allocated organs and patients
    assignment, value = gp.multidict(value_dict_new)

    for (o, p) in value_dict.keys():
        if o in rm_list:
            m.remove(m.getVarByName(f'assign[{o},{p}]'))
            if (o, p) in x:
                del x[o, p]
    # Add new variables for new organs and patients
    for o in new_organs:
        for p in patients:
            x[o, p] = m.addVar(vtype=GRB.BINARY, name=f'assign[{o},{p}]')
    
    for p in new_patients:
        for o in organs:
            x[o, p] = m.addVar(vtype=GRB.BINARY, name=f'assign[{o},{p}]')

    m.update()
    m.addConstrs((x.sum(o, '*') == 1 for o in new_organs), name='organ')
    m.addConstrs((x.sum('*', p) <= 1 for p in new_patients), name='patient')
    if obj == 'envy':
        n = len(patients)
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
    if obj == 'welfare':
        m.setObjective(x.prod(value), GRB.MAXIMIZE)
    elif obj == 'envy':
        m.setObjective(x.prod(value) - (1/n) * x.prod(envy), GRB.MAXIMIZE)
    m.optimize()
    if m.status == GRB.OPTIMAL:
        print("Optimal solution found.")
    elif m.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
        m.computeIIS()
        m.write("model.ilp")
        print("Infeasibility report written to model.ilp")
    elif m.status == GRB.UNBOUNDED:
        print("Model is unbounded.")
    else:
        print("Optimization was stopped with status", m.status)

    return m, x

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
    obj = 'envy'
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


    value_dict = create_value_dict(organs, patients, organ_times, E_times, 
                                   M_times, flag_liver_size=True, alpha=0.5, 
                                   women_indices=women_indices, small_organ_indices=small_organ_indices, 
                                   flag_discrete_t=False)
    
    m, x = allocate_organs(organs, patients, obj, value_dict, flag_print_allocation=False)
    
    n_range = add_T
    available_patients = patients
    available_organs = organs
    for ttime in range((T-init_T)//add_T):
        print('current time: ',n_range)
        #seperate organs became available befor n_range time
        keys = np.fromiter(organ_times.keys(), dtype=float)
        vals = np.fromiter(organ_times.values(), dtype=float)
        organ_range = vals <= n_range
        organ_range = keys[organ_range]
        organ_times = {k: organ_times[k] for k in keys[vals > n_range]}

        # Remove allocated organs and patients
        rm_list = []
        n_organ_all = 0 
        new_organs, new_patients = [], []
        n_organ_all = 0
        for i in available_patients:
            for j in organ_range:
                if x[j, i].x > 0.5:
                    print(f"Patient {i} gets organ {j}")
                    available_patients.remove(i)
                    n_organ_all += 1
        for j in organ_range:
            available_organs.remove(j)
            rm_list.append(j)
        print(f'num organs allocated from {n_range-add_T} until {n_range} is {n_organ_all} out of {len(organ_range)}')
        #Update the patients and organ instances
        new_organ_t, new_E_times, new_w_indices, new_s_organ_indices = gen_data(add_T, add_omega, add_n, init_t=init_T)
        new_patients.extend(range(max(available_patients)+1, max(available_patients)+1+add_n))
        new_organs.extend(range(max(available_organs)+1, max(available_organs)+1+add_omega))
        # Updates input times
        organ_times.update({o: t for o, t in zip(new_organs, new_organ_t)})
        E_times.update({p: t for p, t in zip(new_patients, new_E_times)})
        M_times.update({p: np.random.randint(E_times[p], T) for p in new_patients})
        new_w_indices += init_T
        new_s_organ_indices += init_omega
        #update the model based on new organs and patients 
        available_patients.extend(new_patients)
        available_organs.extend(new_organs)
        value_dict_new = create_value_dict(available_organs, available_patients, organ_times, E_times, 
                                   M_times, flag_liver_size=True, alpha=0.5, 
                                   women_indices=new_w_indices, small_organ_indices=new_s_organ_indices, 
                                   flag_discrete_t=False)
        
        m, x = update_model(m, x, available_organs, available_patients, new_organs, new_patients, value_dict, rm_list, value_dict_new, obj=obj)
        value_dict = value_dict_new
        init_T += add_T
        n_range += add_T
        init_omega += add_omega


