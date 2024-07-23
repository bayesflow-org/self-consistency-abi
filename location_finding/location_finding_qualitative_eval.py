from cmdstanpy import CmdStanModel
from matplotlib import cm
import bayesflow as bf
from matplotlib.colors import LinearSegmentedColormap
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import matplotlib.pyplot as plt

from location_finding_quantitative_eval import load_checkpoint, mmd_vectorized
from tasks.location_finding import get_prior_dist, LocationFinding
import argparse


if __name__ == "__main__":
    # make K an argument and parse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--K", type=int, default=1)
    argparser.add_argument("--sc", type=int, default=5)
    argparser.add_argument("--sc-lambda", type=float, default=0.001)
    argparser.add_argument("--sim-budget", type=int, default=256)
    argparser.add_argument("--seed", type=int, default=3)
    argparser.add_argument("--num-posterior-samples", type=int, default=4096)
    args = argparser.parse_args()

    PLOT_DIR = Path("plots_r")
    ckpt_dir = Path(f"checkpoints/location_finding_K{args.K}_{args.sc_lambda}")

    tf.random.set_seed(args.seed)
    n_posterior_samples = args.num_posterior_samples

    theta_actual_shape = (args.K, 2)
    # test_theta = train_data["prior_draws"][0:1, :]
    prior_dist = get_prior_dist(K=args.K)
    simulator = LocationFinding(K=args.K)
    test_theta = prior_dist.sample(1)  # [1, K*2]

    theta_reshaped = tf.reshape(test_theta, [1, *theta_actual_shape])
    # test_theta = tf.constant([[-1.0, -1, 1, 1]], dtype=tf.float32)
    ys = simulator(test_theta)
    # summary conditions should be the concat of ys and simulator.x
    # summary_conditions = tf.concat([ys, tf.expand_dims(simulator.x, 0)], axis=-1)
    test_sims = {"summary_conditions": ys, "parameters": test_theta}

    ys_np = ys.numpy()  # type: ignore
    print("Simulated dataset shape: ", ys_np.shape)
    y_stan = ys_np[..., [0]][0]
    x_stan = ys_np[..., 1:][0]
    print("y_stan.shape", y_stan.shape)
    print("x_stan.shape", x_stan.shape)
    stan_data = {
        "K": int(args.K),  # Number of sources
        "M": int(simulator.M),  # type: ignore Number of measurement points
        "y": y_stan,  # [M]
        "x": x_stan,  # [M, 2]
    }
    stan_model = CmdStanModel(stan_file="tasks/location_finding.stan")
    fit = stan_model.sample(
        data=stan_data,
        iter_warmup=5000,
        iter_sampling=n_posterior_samples // 16,
        chains=16,
        show_progress=False,
    )

    mcmc_posterior_samples = fit.stan_variable("theta")

    # reshape to [B, K, 2]
    mcmc_posterior_samples_reshaped = np.reshape(
        mcmc_posterior_samples, [n_posterior_samples, *theta_actual_shape]
    )
    mcmc_posterio_samples_flattened = np.reshape(
        mcmc_posterior_samples, [n_posterior_samples, args.K * 2]
    )  # for mmds

    cmap = plt.get_cmap("viridis", 6)
    hmc_color = cmap(4)
    npe_color = cmap(0)
    sc_color = cmap(3)

    f, axes = plt.subplots(1, 3, figsize=(12, 2.2))
    for i in range(theta_actual_shape[0]):
        # plot the mcmc_posterior_samples samples first
        axes[0].scatter(
            mcmc_posterior_samples_reshaped[:, i, 0],
            mcmc_posterior_samples_reshaped[:, i, 1],
            s=1,
            alpha=0.1,
            label="True",
            c=hmc_color,
        )
        axes[0].scatter(
            simulator.x[:, 0],  # type: ignore
            simulator.x[:, 1],  # type: ignore
            c=ys[..., 0],  # type: ignore
            s=10,
            marker="x",
            cmap="GnBu",
            label="Measurements",
        )
        axes[0].scatter(
            theta_reshaped[:, i, 0],
            theta_reshaped[:, i, 1],
            s=100,
            c="red",
            marker="*",
            label="Source locations",
        )

        axes[0].set_title("True", fontsize=18)
        # set limits to [-3 3]
        axes[0].set_xlim(-3, 3)
        axes[0].set_ylim(-3, 3)

    ######### AMORTIZED POST SAMPLES #########
    print("Sampling from amortized posterior...")
    npe_ckpt = ckpt_dir / f"npe_{args.sim_budget}"
    sc_ckpt = ckpt_dir / f"sc{args.sc}_{args.sim_budget}"

    trainers = {
        "NPE": load_checkpoint(npe_ckpt, K=args.K, sc=False),
        "SC": load_checkpoint(sc_ckpt, K=args.K, sc=True),
    }

    for tr_num, (name, trainer) in enumerate(trainers.items()):
        print(f"Sampling from {name}...")
        amortized_samples = trainer.amortizer.sample(
            test_sims, n_samples=n_posterior_samples
        )
        print("amortized_samples shape", amortized_samples.shape)
        print("test_theta", test_theta)

        # compute MMDs
        mmd = mmd_vectorized(
            amortized_samples[np.newaxis, ...],
            mcmc_posterio_samples_flattened[np.newaxis, ...],
        )
        print("mmd", mmd)
        # reshape samples to [B, K, 2]
        samples_reshaped = tf.reshape(
            amortized_samples, [n_posterior_samples, *theta_actual_shape]
        )
        theta_reshaped = tf.reshape(test_theta, [1, *theta_actual_shape])

        cmap = plt.get_cmap("viridis", 6)

        # iterate over K
        for i in range(theta_actual_shape[0]):
            axes[tr_num + 1].scatter(
                samples_reshaped[:, i, 0],
                samples_reshaped[:, i, 1],
                s=1,
                alpha=0.05,
                c=npe_color if name == "NPE" else sc_color,
            )

        for i in range(theta_actual_shape[0]):
            axes[tr_num + 1].scatter(
                theta_reshaped[:, i, 0],
                theta_reshaped[:, i, 1],
                s=100,
                c="red",
                marker="*",
                label="Source locations",
            )
            # plot measurement points  simulator.x, [M, 2]
            # color by ys
            axes[tr_num + 1].scatter(
                simulator.x[:, 0],  # type: ignore
                simulator.x[:, 1],  # type: ignore
                # change color scheme to red-green
                # cmap=LinearSegmentedColormap.from_list(
                #     "red_green", ["red", "green"], N=256
                # ),
                cmap="GnBu",
                c=ys[..., 0],  # type: ignore
                s=10,
                marker="x",
                label="Measurements",
            )
        axes[tr_num + 1].set_title(f"{name} | MMD={round(mmd[0], 3)}", fontsize=16)

        # set limits:
        axes[tr_num + 1].set_xlim(-3, 3)
        axes[tr_num + 1].set_ylim(-3, 3)
    plt.savefig(
        PLOT_DIR / f"locfin_posteriors_K{args.K}_seed{args.seed}.png",
        bbox_inches="tight",
    )
    plt.close()
