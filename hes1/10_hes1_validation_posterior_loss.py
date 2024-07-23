import pickle
from pathlib import Path

import bayesflow as bf
import numpy as np
from bayesflow.computational_utilities import posterior_calibration_error

import matplotlib.pyplot as plt
from scripts.hes1_nple_train import get_trainers
from tasks.hes1 import generative_model
import seaborn as sns

COMPUTATIONS_DIR = Path(__file__).parents[1] / "computations"
PLOT_DIR = Path(__file__).parents[1] / "plots"


def validation_loss_eval():
    trainer_names = ["baseline", "sc", "sc_bad_likelihood_net"]

    simulation_budgets = [128, 256, 512, 1024]
    validation_losses = {name: [] for name in trainer_names}

    for budget in simulation_budgets:
        trainers, names = get_trainers(simulation_budget=budget)

        for i, _ in enumerate(trainer_names):
            validation_loss = (
                trainers[i]
                .loss_history.get_plottable()["val_losses"]["Post.Loss"]
                .iloc[-1]
            )

            validation_losses[trainer_names[i]].append(validation_loss)

    return validation_losses


def validation_loss_plot():
    validation_loss = validation_loss_eval()
    cmap = plt.get_cmap("viridis", 3)
    fig, ax = plt.subplots(1, 1, figsize=(16, 7.5))

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
    simulation_budgets = [128, 256, 512, 1024]
    linestyles = ["solid", "dotted", "dashed"]

    for j, estimator in enumerate(estimators):
        label = estimators[estimator]["label"]
        color = estimators[estimator]["color"]

        ax.plot(
            range(len(simulation_budgets)),
            validation_loss[estimator],
            color=color,
            label=label,
            linestyle=linestyles[j],
            linewidth=5,
        )
        ax.plot(
            range(len(simulation_budgets)),
            validation_loss[estimator],
            marker="o",
            markersize=15,
            color=color,
        )

        ax.set_xticks(range(len(simulation_budgets)))
        ax.tick_params(axis="both", which="major", labelsize=46, length=5, pad=20)
        ax.set_xticklabels(
            [rf"$2^{{{int(l)}}}$" for l in np.log2(np.array(simulation_budgets))]
        )

    sns.despine()
    # fig.supxlabel("Simulation budget", fontsize=54)
    fig.supylabel("Posterior loss\n (validation)", fontsize=54, verticalalignment="center")
    # fig.legend(fontsize=14, loc="upper right")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "hes_1_simulation_budget_validation.pdf")

    return validation_loss


if __name__ == "__main__":
    validation_loss_plot()
