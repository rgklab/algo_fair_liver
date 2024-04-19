from allocation import allocate_organs

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

m = allocate_organs(
    organs, patients, organ_times, E_times, M_times,
    obj='envy', flag_liver_size=True,
    women_indices=[2], small_organ_indices=[1],
    flag_print_allocation=True
)
opt_val = m.getObjective().getValue()
print("Final value is " + str(opt_val))