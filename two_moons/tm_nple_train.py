import logging
import os
import pickle
from pathlib import Path
from typing import Callable

import bayesflow as bf
import tensorflow as tf

from sc_abi.sc_amortizers import AmortizedPosteriorLikelihoodSC
from sc_abi.sc_schedules import ZeroLinearOneSchedule, ZeroOneSchedule
from tasks.two_moons import generative_model, get_amortizer_arguments, prior

tf.random.set_seed(1)

TASK_NAME = "two_moons"

SIMULATED_DATA_DIR = Path("simulated_data", TASK_NAME)
PLOT_DIR = Path("plots", TASK_NAME)
CHECKPOINT_DIR = Path("checkpoints/", TASK_NAME)

if not SIMULATED_DATA_DIR.exists():
    SIMULATED_DATA_DIR.mkdir()

if not PLOT_DIR.exists():
    PLOT_DIR.mkdir()


def main(
    simulation_budget: int = 1024,
    lr: float = 5e-4,
    epochs: int = 200,
    batch_size: int = 32,
    sc_samples: int = 10,
    run_id: int = 0,
    lambda_scheduler: Callable | None = None,
):
    # create training data
    filename = f"{TASK_NAME}_{simulation_budget}_sims.pkl"

    if os.path.exists(SIMULATED_DATA_DIR / filename):
        logging.log(logging.INFO, "Loading training data...")
        with open(SIMULATED_DATA_DIR / filename, "rb") as f:
            train_data = pickle.load(f)
    else:
        logging.log(logging.INFO, "Generating training data...")
        logging.getLogger().setLevel(logging.ERROR)
        train_data = generative_model(simulation_budget)
        with open(SIMULATED_DATA_DIR / filename, "wb") as f:
            pickle.dump(train_data, f)

        # data shapes
        # for key, value in train_data.items():
        #    try:
        #        logging.log(logging.INFO, f"{key}: {value.shape}")
        #    except Exception:
        #        pass #logging.log(logging.INFO, f"{key}: {len(value)}")

    # standard trainer (no self-consistency)
    trainer_nple = bf.trainers.Trainer(
        amortizer=bf.amortizers.AmortizedPosteriorLikelihood(
            **get_amortizer_arguments()
        ),
        generative_model=generative_model,
        default_lr=lr,
        memory=False,
        checkpoint_path=CHECKPOINT_DIR / str(simulation_budget) / "nple" / str(run_id),
        configurator=bf.benchmarks.Benchmark("two_moons", "joint").configurator,
        max_to_keep=1,
        save_checkpoint=False,
    )

    # SC-ABI trainer
    if lambda_scheduler is None:
        lambda_scheduler = ZeroOneSchedule(threshold_step=32 * 100)
    trainer_sc10 = bf.trainers.Trainer(
        amortizer=AmortizedPosteriorLikelihoodSC(
            **get_amortizer_arguments(),
            prior=prior,
            n_consistency_samples=sc_samples,
            lambda_schedule=lambda_scheduler,
            theta_clip_value_min=-2.0,
            theta_clip_value_max=2.0
            - 1e-5,  # uniform distribution excludes upper limit
        ),
        generative_model=generative_model,
        default_lr=lr,
        memory=False,
        checkpoint_path=CHECKPOINT_DIR / str(simulation_budget) / "sc" / str(run_id),
        configurator=bf.benchmarks.Benchmark("two_moons", "joint").configurator,
        max_to_keep=1,
        save_checkpoint=False,
    )
    trainers = {
        "baseline": trainer_nple,
        "sc10": trainer_sc10,
    }
    for name, trainer in trainers.items():
        logging.log(logging.INFO, f"Training {name}...")
        trainer.train_offline(train_data, epochs=epochs, batch_size=batch_size)
        trainer._save_trainer(True)
        tf.keras.backend.clear_session()
        del trainer


if __name__ == "__main__":
    # define an arparser and add sim-budget as an argument
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-budget", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sc-samples", type=int, default=10)
    # type of lambda scheduler: zero-one, zero-linear-one
    parser.add_argument("--lambda-scheduler", type=str, default="zero-one")
    # threshold
    parser.add_argument("--threshold-step", type=int, default=32 * 100)
    parser.add_argument("--lmd", type=float, default=1.0)
    parser.add_argument("--run-id", type=int, default=0)
    args = parser.parse_args()

    # if zero-linear-one threhold 2 is 2*threhosld-step
    if args.lambda_scheduler == "zero-linear-one":
        threshold1 = args.threshold_step
        threshold2 = threshold1 * 2
        lambda_scheduler = ZeroLinearOneSchedule(
            threshold1=threshold1, threshold2=threshold2, lmd=args.lmd
        )
    else:
        lambda_scheduler = ZeroOneSchedule(threshold_step=args.threshold_step)

    main(
        simulation_budget=args.sim_budget,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sc_samples=args.sc_samples,
        lambda_scheduler=lambda_scheduler,
        run_id=args.run_id,
    )
