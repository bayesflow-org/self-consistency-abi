import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

COMPUTATIONS_DIR = Path(__file__).parents[1] / "computations"
PLOT_DIR = Path(__file__).parents[1] / "plots"


def mmd_by_sample_size():
    with open(COMPUTATIONS_DIR / "hes1_eval_dict.pkl", "rb") as f:
        eval_dict = pickle.load(f)

    cmap = plt.get_cmap("viridis", 3)

    simulation_budgets = [64, 128, 256, 512, 1024]
    run_ids = [1]
    estimators = {
        "baseline": {"label": "NPLE (baseline)", "color": cmap(0)},
        "sc": {"label": "SC-NPLE (ours)", "color": cmap(1)},
        # "linear": {"label": "SC-NPLE (linear schedule)", "color": cmap(2)},
        # "betacdf": {"label": "SC-NPLE (betacdf schedule)", "color": cmap(3)},
        "sc_bad_likelihood_net": {
            "label": "SC-NPLE (bad lik. net)",
            "color": cmap(2),
        },
    }

    linestyles = ["solid", "dotted", "dashed"]
    fig, ax = plt.subplots(1, 1, figsize=(16, 7.5))

    for j, estimator in enumerate(estimators.keys()):
        label = estimators[estimator]["label"]
        color = estimators[estimator]["color"]
        mmds = np.array(
            [
                [eval_dict[estimator][budget][run_id]["mmd"] for run_id in run_ids]
                for budget in simulation_budgets
            ]
        )

        mmd_test_mean = np.mean(mmds, axis=2)
        mmd_runs_median = np.median(mmd_test_mean, axis=1)
        mmd_runs_best = np.min(mmd_test_mean, axis=1)
        mmd_runs_worst = np.max(mmd_test_mean, axis=1)

        ax.plot(
            range(len(simulation_budgets)),
            mmd_runs_median,
            color=color,
            label=label,
            linestyle=linestyles[j],
            linewidth=5,
        )

        # ax.fill_between(
        #     range(len(simulation_budgets)), mmd_runs_best, mmd_runs_worst, alpha=0.3, color=color
        # )
        ax.plot(
            range(len(simulation_budgets)), mmd_runs_median, label=label, marker="o", color=color,
            markersize=15, linestyle=linestyles[j]
        )

   # ax.set_xlabel("Simulation budget", fontsize=14)
    fig.supylabel("Posterior MMD", fontsize=54, verticalalignment="center")
    ax.set_ylim(0, None)

    # add gridlines
    ax.set_xticks(range(len(simulation_budgets)))
    sns.despine()

    # larger font for everything
    ax.tick_params(axis="both", which="major", labelsize=46, length=5, pad=20)

    ax.set_xticklabels(
        [rf"$2^{{{int(l)}}}$" for l in np.log2(np.array(simulation_budgets))]
    )

    # fig.legend(fontsize=14, loc="upper right")
    fig.tight_layout()
    fig.subplots_adjust(left=0.20)
    fig.savefig(PLOT_DIR / "hes_1_simulation_budget_mmd.pdf")




if __name__ == "__main__":
    mmd_by_sample_size()
