import logging
import os
import pickle
from pathlib import Path

import bayesflow as bf
from bayesflow.amortizers import (
    AmortizedPosterior,
    AmortizedLikelihood,
    AmortizedPosteriorLikelihood,
)
from bayesflow.networks import InvertibleNetwork

from sc_abi.sc_amortizers import (
    AmortizedPosteriorLikelihoodSC,
)
from sc_abi.sc_schedules import ZeroOneSchedule
from tasks.hes1 import (
    configurator,
    generative_model,
    prior,
    get_latent_dist,
)

SIMULATED_DATA_DIR = Path(__file__).parents[1] / "simulated_data"
PLOT_DIR = Path(__file__).parents[1] / "plots"
CHECKPOINT_DIR = Path(__file__).parents[1] / "checkpoints" / "hes1"

if not SIMULATED_DATA_DIR.exists():
    SIMULATED_DATA_DIR.mkdir()

if not PLOT_DIR.exists():
    PLOT_DIR.mkdir()


def create_training_data(simulation_budget):
    filename = f"hes1_{simulation_budget}_sims.pkl"

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

    return train_data


def get_trainers(simulation_budget=512, run_id=1):
    trainer_baseline = bf.trainers.Trainer(
        amortizer=AmortizedPosteriorLikelihood(
            amortized_posterior=AmortizedPosterior(
                inference_net=InvertibleNetwork(
                    num_params=4,
                    num_coupling_layers=4,
                    coupling_settings={"spec_norm": True},
                    coupling_design="spline",
                ),
                latent_dist=get_latent_dist(num_params=4),
            ),
            amortized_likelihood=AmortizedLikelihood(
                surrogate_net=InvertibleNetwork(
                    num_params=8,
                    num_coupling_layers=4,
                    coupling_settings={"spec_norm": True},
                    coupling_design="spline",
                ),
                latent_dist=get_latent_dist(num_params=8),
            ),
        ),
        generative_model=generative_model,
        default_lr=1e-3,
        configurator=configurator,
        memory=False,
        checkpoint_path=CHECKPOINT_DIR / f"baseline_{simulation_budget}_{run_id}",
        max_to_keep=1,
    )

    threshold_step = {2 ** 6: 51, 2 ** 7: 91, 2 ** 8: 171, 2 ** 9: 331, 2 ** 10: 651, 2 ** 11: 1291, 2 ** 12: 2571}
    trainer_sc = bf.trainers.Trainer(
        amortizer=AmortizedPosteriorLikelihoodSC(
            prior=prior,
            lambda_schedule=ZeroOneSchedule(threshold_step=threshold_step[simulation_budget]),
            n_consistency_samples=100,
            amortized_posterior=AmortizedPosterior(
                inference_net=InvertibleNetwork(
                    num_params=4,
                    num_coupling_layers=4,
                    coupling_settings={"spec_norm": True},
                    coupling_design="spline",
                ),
                latent_dist=get_latent_dist(num_params=4),
            ),
            amortized_likelihood=AmortizedLikelihood(
                surrogate_net=InvertibleNetwork(
                    num_params=8,
                    num_coupling_layers=4,
                    coupling_settings={"spec_norm": True},
                    coupling_design="spline",
                ),
                latent_dist=get_latent_dist(num_params=8),
            ),
        ),
        generative_model=generative_model,
        default_lr=1e-3,
        configurator=configurator,
        memory=False,
        checkpoint_path=CHECKPOINT_DIR / f"sc_{simulation_budget}_{run_id}",
        max_to_keep=1,
    )

    trainer_sc_bad_likelihood_net = bf.trainers.Trainer(
        amortizer=AmortizedPosteriorLikelihoodSC(
            prior=prior,
            lambda_schedule=ZeroOneSchedule(threshold_step=threshold_step[simulation_budget]),
            n_consistency_samples=100,
            amortized_posterior=AmortizedPosterior(
                inference_net=InvertibleNetwork(
                    num_params=4,
                    num_coupling_layers=4,
                    coupling_settings={"spec_norm": True},
                    coupling_design="spline",
                ),
                latent_dist=get_latent_dist(num_params=4),
            ),
            amortized_likelihood=AmortizedLikelihood(
                surrogate_net=InvertibleNetwork(
                    num_params=8,
                    num_coupling_layers=1,
                    coupling_settings={
                        "spec_norm": True,
                        "dense_args": dict(
                            kernel_regularizer=None, activation="linear"
                        ),
                        "dropout": False,
                    },
                    coupling_design="spline",
                ),
                latent_dist=get_latent_dist(num_params=8),
            ),
        ),
        generative_model=generative_model,
        default_lr=1e-3,
        configurator=configurator,
        memory=False,
        checkpoint_path=CHECKPOINT_DIR
                        / f"sc_bad_likelihood_net_{simulation_budget}_{run_id}",
        max_to_keep=1,
    )

    train_data = create_training_data(simulation_budget)
    if trainer_baseline.loss_history.latest == 0:
        trainer_baseline.train_offline(
            train_data, batch_size=16, epochs=70, validation_sims=300
        )

    if trainer_sc.loss_history.latest == 0:
        trainer_sc.train_offline(
            train_data, batch_size=16, epochs=70, validation_sims=300
        )

    if trainer_sc_bad_likelihood_net.loss_history.latest == 0:
        trainer_sc_bad_likelihood_net.train_offline(
            train_data, batch_size=16, epochs=70, validation_sims=300
        )

    names = [
        f"baseline_{simulation_budget}_{run_id}",
        f"sc_{simulation_budget}_{run_id}",
        f"sc_bad_likelihood_net_{simulation_budget}_{run_id}",
    ]

    return [
        trainer_baseline,
        trainer_sc,
        trainer_sc_bad_likelihood_net,
    ], names


if __name__ == "__main__":
    for x in range(7, 13):
        get_trainers(simulation_budget=2 ** x, run_id=2)
