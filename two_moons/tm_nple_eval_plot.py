import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import numpy as np
import seaborn as sns

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=12)

cmap = plt.get_cmap("viridis", 6)

simulation_budgets = [256, 512, 1024, 2048, 4096]
run_ids = [0, 1, 2, 5, 6]
estimators = {
    'nple': {'label': 'NPLE (baseline)',
             'color': cmap(0)},
    'sc': {'label': 'SC-NPLE (ours)',
           'color': cmap(3)},
}

# open eval_dict
with open('./computations/two_moons/eval_dict.pkl', 'rb') as f:
    eval_dict = pickle.load(f)

# plot mmd as a function of simulation budget for each estimator
fig, ax = plt.subplots(1, 1, figsize=(5, 3))

for i, estimator in enumerate(estimators.keys()):
    label = estimators[estimator]['label']
    color = estimators[estimator]['color']
    mmds = np.array([[eval_dict[estimator][budget][run_id]['mmd'] 
                     for run_id in run_ids] 
                     for budget in simulation_budgets])
    mmd_test_mean = np.mean(mmds, axis=2)
    mmd_runs_median = np.median(mmd_test_mean, axis=1)
    mmd_runs_best = np.min(mmd_test_mean, axis=1)
    mmd_runs_worst = np.max(mmd_test_mean, axis=1)
    ax.fill_between(simulation_budgets, mmd_runs_best, mmd_runs_worst, alpha=0.3, color=color)
    ax.plot(simulation_budgets, mmd_runs_median, label=label, marker='o', color=color)
    # ax.errorbar(np.array(simulation_budgets)-15+30*i, mmd_runs_mean, yerr=mmd_runs_std, label=label, marker='o', capsize=5, color=color)

ax.set_xlabel('Simulation budget', fontsize='xx-large')
ax.set_ylabel('Posterior MMD', fontsize='xx-large')
ax.set_ylim(0, None)
ax.set_yticks([0, 0.10, 0.20], minor=False)
ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20], minor=True)
# add gridlines
ax.grid(axis='y', linestyle='-', which="major", alpha=0.4)
ax.grid(axis='y', linestyle='-', which="minor", alpha=0.15)
ax.grid(axis='x', linestyle='-', alpha=0.5)
ax.set_xticks(simulation_budgets)
sns.despine()

# larger font for everything
ax.tick_params(axis='both', which='major', labelsize='x-large')

# ax.set_xticklabels([fr"$2^{{{int(l)}}}$" for l in np.log2(np.array(simulation_budgets))])
# 45 degree rotation
ax.set_xticklabels(simulation_budgets, rotation=45, ha='center')

# ax.set_title("Two Moons: approximate vs. reference posterior")
ax.legend(fontsize='x-large', loc='upper right')
fig.tight_layout()
fig.savefig('./plots/tm_simulation_budget_mmd.pdf')
sns.despine()
plt.show()

