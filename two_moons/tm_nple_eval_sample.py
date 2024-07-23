from tasks.two_moons import generative_model, prior, get_amortizer_arguments
from sc_abi.sc_amortizers import AmortizedPosteriorLikelihoodSC
from sc_abi.sc_schedules import ZeroOneSchedule, ZeroLinearOneSchedule
import bayesflow as bf
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import logging
from pathlib import Path
import numpy as np
from tasks.two_moons import analytic_posterior_numpy

import pickle

estimators = ['nple', 'sc']
simulation_budgets = [256, 512, 1024, 2048, 4096]
run_ids = [0, 1, 2, 5, 6]

eval_dict = {estimator: {budget: {run_id: {} for run_id in run_ids} for budget in simulation_budgets} for estimator in estimators}

n_eval_instances = 100
n_posterior_draws = 1000

TASK_NAME = "two_moons"

os.makedirs(f'./computations/{TASK_NAME}', exist_ok=True)
SIMULATED_DATA_DIR = Path("simulated_data", TASK_NAME)
PLOT_DIR = Path("plots", TASK_NAME)
CHECKPOINT_DIR = Path("checkpoints/", TASK_NAME)

# Load or generate evaluation data
filename = f"{TASK_NAME}_eval_sims.pkl"

if os.path.exists(SIMULATED_DATA_DIR / filename):
    logging.log(logging.INFO, "Loading eval data...")
    with open(SIMULATED_DATA_DIR / filename, "rb") as f:
        eval_data = pickle.load(f)
else:
    logging.log(logging.INFO, "Generating eval data...")
    logging.getLogger().setLevel(logging.ERROR)
    eval_data = generative_model(n_eval_instances)
    with open(SIMULATED_DATA_DIR / filename, "wb") as f:
        pickle.dump(eval_data, f)


# compute analytic posterior (ground-truth)
y_eval = eval_data['sim_data']
theta_eval = eval_data['prior_draws']

n_sim, n_params = theta_eval.shape

analytic_samples = np.empty((n_sim, n_posterior_draws, n_params))

for i in range(n_sim):
    analytic_samples[i] = analytic_posterior_numpy(y_eval[i], n_posterior_draws, rng=np.random.default_rng(seed=1234))
analytic_samples = analytic_samples.astype(np.float32)

# Evaluation
for simulation_budget in simulation_budgets:
    for run_id in run_ids:
        # standard trainer (no self-consistency)
        trainer_nple = bf.trainers.Trainer(
            amortizer=bf.amortizers.AmortizedPosteriorLikelihood(**get_amortizer_arguments()),
            generative_model=generative_model,
            default_lr=5e-4,
            memory=False,
            checkpoint_path=CHECKPOINT_DIR / str(simulation_budget) / "nple" / str(run_id),
            configurator=bf.benchmarks.Benchmark('two_moons', 'joint').configurator,
            max_to_keep=1,
        )
        eval_data_configured = trainer_nple.configurator(eval_data)
        
        
        posterior_samples = trainer_nple.amortizer.sample_parameters(eval_data_configured, n_posterior_draws)
        mmd = np.array([
            bf.computational_utilities.maximum_mean_discrepancy(analytic_samples[i], posterior_samples[i])
            for i in range(n_eval_instances)])
        
        eval_dict['nple'][simulation_budget][run_id]['posterior_samples'] = posterior_samples
        eval_dict['nple'][simulation_budget][run_id]['mmd'] = mmd
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

        posterior_samples = trainer_sc10.amortizer.sample_parameters(eval_data_configured, n_posterior_draws)
        mmd = np.array([
            bf.computational_utilities.maximum_mean_discrepancy(analytic_samples[i].astype(np.float32), posterior_samples[i])
            for i in range(n_eval_instances)])
        
        eval_dict['sc'][simulation_budget][run_id]['posterior_samples'] = posterior_samples
        eval_dict['sc'][simulation_budget][run_id]['mmd'] = mmd
        tf.keras.backend.clear_session()

with open(f'./computations/{TASK_NAME}/eval_dict.pkl', 'wb') as f:
    pickle.dump(eval_dict, f)


