import logging
from pathlib import Path

import bayesflow as bf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.hes1_nple_train import get_trainers
from tasks.hes1 import generative_model, mmd_vectorized, sample_hmc

PLOT_DIR = Path(__file__).parents[1] / "plots"


def stan_comparison():
    logger = logging.getLogger("cmdstanpy")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)

    trainers, names = get_trainers()
    test_sims = trainers[0].configurator(generative_model(100))

    hmc_posterior_samples = sample_hmc(test_sims)
    posterior_samples = [
        trainer.amortizer.sample_parameters(test_sims, 2000) for trainer in trainers
    ]
    df_mmd = pd.DataFrame()
    for i, ps in enumerate(posterior_samples):
        mmd = mmd_vectorized(ps, hmc_posterior_samples)
        df_mmd[names[i]] = mmd

    # mmd histogram
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.set_yticks(np.arange(0, 4, 0.5), minor=False)
    ax.set_yticks(np.arange(0, 4, 0.1), minor=True)
    ax.grid(axis="y", which="major", alpha=0.6)
    ax.grid(axis="y", which="minor", alpha=0.15)
    ax.set_ylabel("Posterior MMD")
    ax.set_xlabel("Method")
    sns.boxplot(
        data=df_mmd,
        ax=ax,
        width=0.4,
        flierprops={"marker": "."},
        boxprops={"facecolor": (0.50, 0, 0.0, 0.3)},
        linewidth=1,
    )

    ax.tick_params(axis="both", which="both")
    sns.despine()
    plt.ylim(0, None)
    fig.savefig(PLOT_DIR / "hes1_mmd_boxplot.pdf", bbox_inches="tight")

    print(df_mmd.mean())

    posterior_samples = [hmc_posterior_samples, *posterior_samples]
    names = ["hmc", *names]

    for i, posterior_samples in enumerate(posterior_samples):
        f = bf.diagnostics.plot_sbc_ecdf(
            posterior_samples,
            test_sims["posterior_inputs"]["parameters"],
            difference=True,
        )
        f.savefig(PLOT_DIR / f"hes1_{names[i]}_sbc.pdf")

        f = bf.diagnostics.plot_recovery(
            posterior_samples, test_sims["posterior_inputs"]["parameters"]
        )
        f.savefig(PLOT_DIR / f"hes1_{names[i]}_recovery.pdf")


if __name__ == "__main__":
    stan_comparison()
