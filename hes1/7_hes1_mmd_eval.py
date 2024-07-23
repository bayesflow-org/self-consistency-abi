import pickle
from pathlib import Path

import bayesflow as bf
from bayesflow.amortizers import (
    AmortizedLikelihood,
    AmortizedPosterior,
    AmortizedPosteriorLikelihood,
)
from bayesflow.networks import InvertibleNetwork

from sc_abi.sc_amortizers import (
    AmortizedPosteriorLikelihoodSC,
)
from sc_abi.sc_schedules import BetaCDFSchedule, LinearSchedule, ZeroOneSchedule
from scripts.hes1_nple_train import get_trainers
from tasks.hes1 import (
    configurator,
    generative_model,
    get_latent_dist,
    mmd_vectorized,
    prior,
    sample_hmc,
)

CHECKPOINT_DIR = Path(__file__).parents[1] / "checkpoints" / "hes1"
SIMULATED_DATA_DIR = Path(__file__).parents[1] / "simulated_data"
COMPUTATIONS_DIR = Path(__file__).parents[1] / "computations"


def eval_dict():
    simulation_budgets = [64, 128, 256, 512, 1024]
    trainer_names = [
        "baseline",
        "sc",
        "sc_bad_likelihood_net",
    ]
    run_ids = [1]

    d = {
        name: {budget: {id: {} for id in run_ids} for budget in simulation_budgets}
        for name in trainer_names
    }

    test_sims, hmc_posterior_samples = hmc_runs()

    for simulation_budget in simulation_budgets:
        for run_id in run_ids:
            trainers, _ = get_trainers(
                simulation_budget=simulation_budget, run_id=run_id
            )
            # compute analytic posterior (ground-truth)

            for i, trainer in enumerate(trainers):
                posterior = trainer.amortizer.sample_parameters(test_sims, 2000)
                mmd = mmd_vectorized(posterior, hmc_posterior_samples)

                d[trainer_names[i]][simulation_budget][run_id][
                    "posterior_samples"
                ] = posterior
                d[trainer_names[i]][simulation_budget][run_id]["mmd"] = mmd

    with open(COMPUTATIONS_DIR / "hes1_eval_dict.pkl", "wb") as f:
        pickle.dump(d, f)

    return d


def hmc_runs():
    hmc_savefile = Path(__file__).parents[1] / "checkpoints" / "hes1" / "hes1_hmc.pkl"
    test_savefile = COMPUTATIONS_DIR / "hes1_test_sims.pkl"

    if not hmc_savefile.exists():
        trainers, _ = get_trainers()
        test_sims = trainers[0].configurator(generative_model(200))
        hmc_posterior_samples = sample_hmc(test_sims)

        with open(hmc_savefile, "wb") as f:
            pickle.dump(hmc_posterior_samples, f)
        with open(test_savefile, "wb") as f:
            pickle.dump(test_sims, f)

    else:
        with open(hmc_savefile, "rb") as f:
            hmc_posterior_samples = pickle.load(f)
        with open(test_savefile, "rb") as f:
            test_sims = pickle.load(f)

    return test_sims, hmc_posterior_samples


if __name__ == "__main__":
    hmc_runs()
    eval_dict()
