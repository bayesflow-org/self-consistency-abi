import os
import tensorflow as tf
import bayesflow as bf
from pathlib import Path
import pickle
import logging

from sc_abi.sc_amortizers import AmortizedPosteriorLikelihoodSC
from sc_abi.sc_schedules import ZeroOneSchedule
from tasks.two_moons import generative_model, prior, get_amortizer_arguments

tf.random.set_seed(1)

SIMULATION_BUDGET = 1024
SIMULATED_DATA_DIR = Path("simulated_data")
PLOT_DIR = Path("plots")
CHECKPOINT_DIR = Path("checkpoints/two_moons")

if not SIMULATED_DATA_DIR.exists():
    SIMULATED_DATA_DIR.mkdir()

if not PLOT_DIR.exists():
    PLOT_DIR.mkdir()

# create training data
filename = f"tm_{SIMULATION_BUDGET}_sims.pkl"

if os.path.exists(SIMULATED_DATA_DIR / filename):
    logging.log(logging.INFO, "Loading training data...")
    with open(SIMULATED_DATA_DIR / filename, "rb") as f:
        train_data = pickle.load(f)
else:
    logging.log(logging.INFO, "Generating training data...")
    logging.getLogger().setLevel(logging.ERROR)
    train_data = generative_model(SIMULATION_BUDGET)
    with open(SIMULATED_DATA_DIR / filename, "wb") as f:
        pickle.dump(train_data, f)

validation_data = generative_model(200)

num_params = train_data["prior_draws"].shape[1]  # train_data is [num_sims, num_params]


# standard trainer (no self-consistency)

trainer_baseline = bf.trainers.Trainer(
    amortizer=bf.amortizers.AmortizedPosteriorLikelihood(**get_amortizer_arguments()),
    generative_model=generative_model,
    default_lr=5e-4,
    memory=False,
    checkpoint_path=CHECKPOINT_DIR / "npe",
    configurator=bf.benchmarks.Benchmark('two_moons', 'joint').configurator,
    max_to_keep=1,
)

trainer_sc = bf.trainers.Trainer(
    amortizer=AmortizedPosteriorLikelihoodSC(
        **get_amortizer_arguments(),
        prior=prior,
        n_consistency_samples=10,
        lambda_schedule=ZeroOneSchedule(threshold_step=32*100),
        theta_clip_value_min=-2.0,
        theta_clip_value_max=2.0,
        ),
    generative_model=generative_model,
    default_lr=5e-4,
    memory=False,
    checkpoint_path=CHECKPOINT_DIR / "sc",
    configurator=bf.benchmarks.Benchmark('two_moons', 'joint').configurator,
    max_to_keep=1,
)

for trainer in [trainer_baseline, trainer_sc]:    
    trainer.train_offline(train_data, 200, batch_size=32, use_autograph=True, validation_sims=validation_data)
