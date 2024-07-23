import argparse
import tensorflow_probability as tfp
import bayesflow as bf
import logging
import os
import tensorflow as tf

from pathlib import Path
import pickle

from tasks.location_finding import (
    get_prior_dist,
    get_amortizer_arguments,
    PriorLogProb,
    LocationFinding,
)
from sc_abi.sc_amortizers import AmortizedPosteriorSC
from sc_abi.sc_schedules import ZeroOneSchedule, ZeroLinearOneSchedule

SIMULATED_DATA_DIR = Path("simulated_data")
PLOT_DIR = Path("plots")
if not SIMULATED_DATA_DIR.exists():
    SIMULATED_DATA_DIR.mkdir()

if not PLOT_DIR.exists():
    PLOT_DIR.mkdir()


def main(
    lr: float,
    sim_budget: int,
    sc: int = 10,
    epochs: int = 100,
    batch_size: int = 32,
    K: int = 2,
    sc_lambda: float = 0.001,
    anneal_sc_lambda: bool = False,
    ckpt_dir: Path = Path("checkpoints"),
):
    prior = PriorLogProb(get_prior_dist(K=K))
    simulator = LocationFinding(K=K)

    generative_model = bf.simulation.GenerativeModel(  # type: ignore
        prior=prior,
        simulator=simulator,
        prior_is_batched=True,
        simulator_is_batched=True,
    )
    # create training data
    filename = f"locfin_k{K}_{sim_budget}_sims.pkl"

    if os.path.exists(SIMULATED_DATA_DIR / filename):
        logging.log(logging.INFO, "Loading training data...")
        with open(SIMULATED_DATA_DIR / filename, "rb") as f:
            train_data = pickle.load(f)
    else:
        logging.log(logging.INFO, "Generating training data...")
        logging.getLogger().setLevel(logging.ERROR)
        train_data = generative_model(sim_budget)
        with open(SIMULATED_DATA_DIR / filename, "wb") as f:
            pickle.dump(train_data, f)

    # standard trainer (no self-consistency)
    trainer_npe = bf.trainers.Trainer(
        amortizer=bf.amortizers.AmortizedPosterior(**get_amortizer_arguments(K=K)),
        generative_model=generative_model,
        default_lr=lr,
        memory=False,
        checkpoint_path=ckpt_dir / f"npe_{sim_budget}",
        max_to_keep=1,
    )

    # threshold 1 should be around epoch 20
    # threshold 2 should be around epoch 80
    threshold1 = int(epochs * sim_budget / batch_size * 0.2)
    if anneal_sc_lambda:
        threshold2 = int(epochs * sim_budget / batch_size * 0.5)
    else:
        threshold2 = threshold1 + 1
    print(f"threshold1: {threshold1}, threshold2: {threshold2}")
    print("totoal gradient steps:", epochs * sim_budget / batch_size)

    # SC-ABI trainer
    trainer_sc = bf.trainers.Trainer(
        amortizer=AmortizedPosteriorSC(
            **get_amortizer_arguments(K=K),
            prior=prior,
            simulator=simulator,
            n_consistency_samples=sc,
            # lambda_schedule=ZeroOneSchedule(threshold_step=1000),
            lambda_schedule=ZeroLinearOneSchedule(  # type: ignore
                threshold1=threshold1, threshold2=threshold2, lmd=sc_lambda
            ),
            theta_clip_value_min=-5.0,
            theta_clip_value_max=5.0,
        ),
        generative_model=generative_model,
        default_lr=lr,
        memory=False,
        checkpoint_path=ckpt_dir / f"sc{sc}_{sim_budget}",
        max_to_keep=1,
    )
    trainers = {
        "sc": trainer_sc,
        "npe": trainer_npe,
    }
    validation_data = generative_model(200)
    for name, trainer in trainers.items():
        print(f"Training {name}...")
        trainer.train_offline(
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_sims=validation_data,
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--K", type=int, default=1)
    argparser.add_argument("--sc-lambda", type=float, default=0.001)
    argparser.add_argument("--sim-budget", type=int, default=256)
    argparser.add_argument("--lr", type=float, default=1e-3)
    argparser.add_argument("--sc", type=int, default=5)
    argparser.add_argument("--epochs", type=int, default=35)
    argparser.add_argument("--batch-size", type=int, default=32)
    argparser.add_argument("--seed", type=int, default=202401)
    args = argparser.parse_args()

    tf.random.set_seed(args.seed)
    ckpt_dir = Path(f"checkpoints/location_finding_K{args.K}_{args.sc_lambda}")

    main(
        sim_budget=args.sim_budget,
        lr=args.lr,
        sc=args.sc,
        epochs=args.epochs,
        batch_size=args.batch_size,
        K=args.K,
        sc_lambda=args.sc_lambda,
        ckpt_dir=ckpt_dir,
    )
