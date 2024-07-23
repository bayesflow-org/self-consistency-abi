import argparse
from matplotlib import cm
import tensorflow as tf
import bayesflow as bf
import numpy as np
import pickle


from pathlib import Path
from tqdm import tqdm

from sc_abi.sc_amortizers import AmortizedPosteriorSC
from sc_abi.sc_schedules import ZeroOneSchedule
from cmdstanpy import CmdStanModel


from tasks.location_finding import (
    get_prior_dist,
    get_amortizer_arguments,
    PriorLogProb,
    LocationFinding,
)

PLOT_DIR = Path("plots_r")


def load_checkpoint(
    checkpoint_name, K: int, sc: bool = False, validate_checkpoint: bool = True
):
    if validate_checkpoint:
        assert checkpoint_name.exists(), f"{checkpoint_name} does not exist."

    # checkpoint_name is path to the checkpoint
    prior = PriorLogProb(get_prior_dist(K=K))
    simulator = LocationFinding(K=K)

    generative_model = bf.simulation.GenerativeModel(  # type: ignore
        prior=prior,
        simulator=simulator,
        prior_is_batched=True,
        simulator_is_batched=True,
    )

    # standard trainer (no self-consistency)
    if sc:
        # SC-ABI trainer
        trainer = bf.trainers.Trainer(
            amortizer=AmortizedPosteriorSC(
                **get_amortizer_arguments(K=K),
                prior=prior,
                simulator=simulator,
                n_consistency_samples=10,
                lambda_schedule=ZeroOneSchedule(threshold_step=200),  # type: ignore
                theta_clip_value_min=-2.0,
                theta_clip_value_max=2.0,
            ),
            generative_model=generative_model,
            default_lr=1e-3,
            memory=False,
            checkpoint_path=checkpoint_name,
            max_to_keep=1,
        )
    else:
        trainer = bf.trainers.Trainer(
            amortizer=bf.amortizers.AmortizedPosterior(**get_amortizer_arguments(K=K)),
            generative_model=generative_model,
            default_lr=1e-3,
            memory=False,
            checkpoint_path=checkpoint_name,
            max_to_keep=1,
        )
    return trainer


def sample_hmc(K: int, test_sims, iter_warmup=2000, iter_sampling=4000, **kwargs):
    summary_conditions = test_sims["summary_conditions"]
    if not isinstance(summary_conditions, np.ndarray):
        summary_conditions = summary_conditions.numpy()
    y_stan = summary_conditions[..., [0]]
    x_stan = summary_conditions[..., 1:]

    n_sim, n_obs, data_dim = x_stan.shape  # [B, 30, 2]

    n_chains = 8
    iter_sampling_per_chain = iter_sampling // n_chains
    posterior_samples = np.zeros((n_sim, iter_sampling, K, 2))

    for i in tqdm(range(n_sim), desc="HMC running on data set"):
        stan_data = {
            "K": int(K),  # Number of sources
            "M": n_obs,  # Number of measurement points
            "y": y_stan[i],  # [M]
            "x": x_stan[i],  # [M, 2]
        }
        model = CmdStanModel(stan_file="tasks/location_finding.stan")
        fit = model.sample(
            data=stan_data,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling_per_chain,
            chains=n_chains,
            show_progress=False,
            **kwargs,
        )
        posterior_samples_chain = fit.stan_variable("theta")
        posterior_samples[i] = posterior_samples_chain

    return posterior_samples


def mmd_vectorized(samples_1, samples_2):
    n_sim = samples_1.shape[0]
    print(samples_1.shape)
    print(samples_2.shape)
    assert n_sim == samples_2.shape[0]
    mmd = np.empty([n_sim])
    for i in tqdm(range(n_sim)):
        mmd_value = bf.computational_utilities.maximum_mean_discrepancy(  # type: ignore
            samples_1[i].astype(np.float32), samples_2[i].astype(np.float32)
        ).numpy()
        mmd[i] = mmd_value
    return mmd


def main(
    K: int,
    num_datasets: int = 50,
    num_posterior_samples: int = 500,
    sim_budgets: list[int] = [4096, 2048, 1024, 512, 256],
    scs: list[int] = [5, 10, 20, 50, 100, 500],
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
    # HMC
    _trainer = load_checkpoint(ckpt_dir / f"npe_{sim_budgets[0]}", K=K, sc=False)
    test_sims = _trainer.configurator(generative_model(num_datasets))

    posterior_samples = sample_hmc(
        K=K, test_sims=test_sims, iter_warmup=2000, iter_sampling=num_posterior_samples
    )
    print("posterior_samples.shape", posterior_samples.shape)
    # flatten the last two dimensions:
    # [B, iter_sampling, K, 2] -> [B, iter_sampling, K * 2]
    posterior_samples_flattened = np.reshape(
        posterior_samples, [num_datasets, num_posterior_samples, K * 2]
    )

    mmds = {}
    # for method, all_checkpoints in {
    #     "NPE": [f"npe_{sim_budget}" for sim_budget in sim_budgets],
    #     "SC": [f"sc{sc}_{sim_budget}" for sim_budget in sim_budgets],
    # }.items():
    mmds = {"npe": {}}

    for sc in scs:
        mmds[f"sc{sc}"] = {}

    for budget in sim_budgets:
        # run npe
        print("npe", budget)
        ckpt_name = ckpt_dir / f"npe_{budget}"
        trainer_npe = load_checkpoint(ckpt_name, K=K, sc=False)
        samples_npe = trainer_npe.amortizer.sample(
            test_sims, n_samples=num_posterior_samples
        )
        mmd = mmd_vectorized(posterior_samples_flattened, samples_npe)
        mmds["npe"][budget] = mmd

        for sc in scs:
            ckpt_name = ckpt_dir / f"sc{sc}_{budget}"
            print(ckpt_name)
            trainer_sc = load_checkpoint(ckpt_name, K=K, sc=True)
            samples_sc = trainer_sc.amortizer.sample(
                test_sims, n_samples=num_posterior_samples
            )
            mmd = mmd_vectorized(
                posterior_samples_flattened, samples_sc
            )  # list of mmds
            mmds[f"sc{sc}"][budget] = mmd
            print(mmds)
            # dump the mmds in the ckp_dir
            with open(
                ckpt_dir / f"mmd_{num_datasets}_{num_posterior_samples}.pkl", "wb"
            ) as f:
                pickle.dump(mmds, f)

    # print averages
    for method, mmds in mmds.items():
        for budget, mmd in mmds.items():
            print(f"{method},{budget}: {np.mean(mmd)}")


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--K", type=int, default=1)
    argparser.add_argument("--sim-budgets", type=int, nargs="+", default=[256])
    argparser.add_argument("--scs", type=int, nargs="+", default=[5])
    argparser.add_argument("--sc-lambda", type=float, default=0.001)
    argparser.add_argument("--latent-df", type=int, default=30)
    argparser.add_argument("--num-datasets", type=int, default=100)
    argparser.add_argument("--num-posterior-samples", type=int, default=512)
    argparser.add_argument("--seed", type=int, default=20240322)
    args = argparser.parse_args()

    ckpt_dir = Path(
        f"checkpoints/location_finding_K{args.K}_{args.sc_lambda}_df{args.latent_df}"
    )
    tf.random.set_seed(args.seed)
    main(
        K=args.K,
        num_datasets=args.num_datasets,
        num_posterior_samples=args.num_posterior_samples,
        scs=args.scs,
        sim_budgets=args.sim_budgets,
        ckpt_dir=ckpt_dir,
    )
