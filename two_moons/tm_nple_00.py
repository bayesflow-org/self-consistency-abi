import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import numpy as np
import seaborn as sns
import os
import bayesflow as bf
import tensorflow as tf
from sc_abi.sc_amortizers import AmortizedPosteriorLikelihoodSC
from sc_abi.sc_schedules import ZeroOneSchedule
from tasks.two_moons import generative_model, prior, get_amortizer_arguments

from tasks.two_moons import analytic_posterior_numpy

cmap = plt.get_cmap("viridis", 6)

simulation_budgets = [512, 1024, 2048, 4096]
run_id = 1
estimators = {
    'nple': {'label': 'NPLE (baseline)',
             'color': cmap(0)},
    'sc': {'label': 'SC-NPLE (ours)',
           'color': cmap(3)},
}


n_posterior_draws = 1000

TASK_NAME = "two_moons"

os.makedirs(f'./computations/{TASK_NAME}', exist_ok=True)
PLOT_DIR = Path("plots", TASK_NAME)
CHECKPOINT_DIR = Path("checkpoints/", TASK_NAME)


if True:
    eval_dict = {estimator: {budget: {} for budget in simulation_budgets} for estimator in estimators}

    sim_data = {'direct_conditions': np.array([[0, 0]]).astype(np.float32)}
    reference_posterior = analytic_posterior_numpy(sim_data['direct_conditions'][0], n_posterior_draws, rng=np.random.default_rng(seed=1234))
    eval_dict['reference_posterior'] = reference_posterior
    for simulation_budget in simulation_budgets:
        trainer_nple = bf.trainers.Trainer(
                amortizer=bf.amortizers.AmortizedPosteriorLikelihood(**get_amortizer_arguments()),
                generative_model=generative_model,
                default_lr=5e-4,
                memory=False,
                checkpoint_path=CHECKPOINT_DIR / str(simulation_budget) / "nple" / str(run_id),
                configurator=bf.benchmarks.Benchmark('two_moons', 'joint').configurator,
                max_to_keep=1,
            )
        
        
        posterior_samples = trainer_nple.amortizer.sample_parameters(sim_data, n_posterior_draws)
        mmd = bf.computational_utilities.maximum_mean_discrepancy(reference_posterior, posterior_samples)
        
        eval_dict['nple'][simulation_budget]['posterior_samples'] = posterior_samples
        eval_dict['nple'][simulation_budget]['mmd'] = mmd
        tf.keras.backend.clear_session()

        # SC-ABI trainer
        lambda_scheduler = ZeroOneSchedule(threshold_step=32*100)
        trainer_sc10 = bf.trainers.Trainer(
            amortizer=AmortizedPosteriorLikelihoodSC(
                **get_amortizer_arguments(),
                prior=prior,
                n_consistency_samples=10,
                lambda_schedule=lambda_scheduler,
                theta_clip_value_min=-2.0,
                theta_clip_value_max=2.0 - 1e-5, # uniform distribution excludes upper limit
                ),
            generative_model=generative_model,
            default_lr=5e-4,
            memory=False,
            checkpoint_path=CHECKPOINT_DIR / str(simulation_budget) / "sc" / str(run_id),
            configurator=bf.benchmarks.Benchmark('two_moons', 'joint').configurator,
            max_to_keep=1,
        )

        posterior_samples = trainer_sc10.amortizer.sample_parameters(sim_data, n_posterior_draws)
        mmd = bf.computational_utilities.maximum_mean_discrepancy(reference_posterior, posterior_samples)
        
        eval_dict['sc'][simulation_budget]['posterior_samples'] = posterior_samples
        eval_dict['sc'][simulation_budget]['mmd'] = mmd
        tf.keras.backend.clear_session()

    with open(f'./computations/{TASK_NAME}/eval_dict_00.pkl', 'wb') as f:
        pickle.dump(eval_dict, f)


with open(f'./computations/{TASK_NAME}/eval_dict_00.pkl', 'rb') as f:
    eval_dict = pickle.load(f)


# posterior draws plotting
run_id = 0
nrows = 2
ncols = len(simulation_budgets) + 1
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2), subplot_kw=dict(box_aspect=1), squeeze=False)

scatter_kws = {
    "alpha": 0.20,
    "rasterized": True,
    "s": 0.7,
    "marker": "D",
}
axes[0, 0].set_title("Reference", size='xx-large')

for ax in axes.flat:
    ax.scatter(eval_dict['reference_posterior'][:, 0],
                     eval_dict['reference_posterior'][:, 1],
                     color=(0.8, 0.4, 0.4),
                     **scatter_kws)

for i, simulation_budget in enumerate(simulation_budgets, 1):
    axes[0, i].set_title(fr"$N={simulation_budget}$", size='xx-large')
    axes[0, i].scatter(eval_dict['nple'][simulation_budget]['posterior_samples'][:, 0],
                       eval_dict['nple'][simulation_budget]['posterior_samples'][:, 1],
                       color='white',
                       **scatter_kws)
    axes[1, i].scatter(eval_dict['sc'][simulation_budget]['posterior_samples'][:, 0],
                          eval_dict['sc'][simulation_budget]['posterior_samples'][:, 1],
                          color='white',
                          **scatter_kws)
    for j, estimator in zip([0, 1], ['nple', 'sc']):
        axes[j, i].annotate(text=fr'MMD={eval_dict[estimator][simulation_budget]["mmd"]:.3f}', 
                    xy=(0.50, 0.85), 
                    xycoords='axes fraction',
                    horizontalalignment='center',
                    fontsize='large',
                    color="white"
                    )


for ax in axes.flat:
    ax.grid(False)
    ax.set_facecolor((0 / 255, 32 / 255, 64 / 255, 1.0))
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.spines["bottom"].set_alpha(0.0)
    ax.spines["top"].set_alpha(0.0)
    ax.spines["right"].set_alpha(0.0)
    ax.spines["left"].set_alpha(0.0)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')

plt.savefig(f'./plots/{TASK_NAME}/posterior_draws_00.pdf', bbox_inches='tight')
plt.show()