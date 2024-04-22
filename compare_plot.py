import numpy as np
import matplotlib.pyplot as plt
import pickle

T = 800
with open(f"results/envy_w_envy_t_{T}", "rb") as fp: 
    envy_obj_list = pickle.load(fp)
with open(f"results/envy_w_welfare_t_{T}", "rb") as fp: 
    welfare_obj_list = pickle.load(fp)
with open(f"results/envy_t_{T}", "rb") as fp:
    ttime = pickle.load(fp)

plt.figure(figsize=(10, 6))
plt.plot(ttime, envy_obj_list, label='with envy objective')
plt.plot(ttime, welfare_obj_list, label='with welfare objective')

plt.title(f'Envy value trained with or without envy objective')
plt.xlabel('time')
plt.ylabel('Envy value')
#plt.grid(True)
plt.legend()
plt.savefig(f'figures/compare_t_{T}.jpg')

plt.show()