from pathlib import Path

import bayesflow as bf
import numpy as np
import pandas as pd
import seaborn as sns
from scripts.hes1_nple_train import get_trainers
from tasks.hes1 import Hes1Dataset, sample_hmc


PLOT_DIR = Path(__file__).parents[1] / "plots"


def run_real_data():
    trainers, names = get_trainers()
    trainer_names = ["baseline", "sc", "sc_bad_likelihood_net"]
    method_names = ["NPLE", "SC-NPLE", "SC-NPLE (bad lik. net)"]
    sim_obs = {
        "posterior_inputs": {
            "direct_conditions": Hes1Dataset().data.values.astype(np.float32)[
                np.newaxis, :
            ]
        }
    }
    hmc_posterior_samples_real = sample_hmc(
        sim_obs, iter_warmup=2000, iter_sampling=500
    )[0]
    posterior_samples = [
        trainers[idx].amortizer.sample_parameters(sim_obs, 2000) for idx in [0, 1, 2]
    ]

    df_mmd = pd.DataFrame()
    dfs = []
    for i, ps in enumerate(posterior_samples):
        mmd = bf.computational_utilities.maximum_mean_discrepancy(
            ps.astype(np.float32),
            hmc_posterior_samples_real.astype(np.float32),
        ).numpy()
        df_mmd[names[i]] = mmd
        dfs.append(pd.DataFrame(ps))
        dfs[-1]["method"] = method_names[i]

    dfs.append(pd.DataFrame(hmc_posterior_samples_real))
    dfs[-1]["method"] = "HMC"

    g = sns.PairGrid(pd.concat(dfs), hue="method")
    g.map_upper(sns.scatterplot, rasterized=True)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot)
    g.add_legend()

    g.savefig(PLOT_DIR / "hes1_real_data_pairgrid.pdf")


if __name__ == "__main__":
    run_real_data()
