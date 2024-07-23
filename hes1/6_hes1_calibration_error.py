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

if not COMPUTATIONS_DIR.is_dir():
    COMPUTATIONS_DIR.mkdir()


def calibration_error_eval():
    cmap = plt.get_cmap("viridis", 5)

    trainer_names = ["baseline", "sc", "sc_bad_likelihood_net"]

    simulation_budgets = [128, 256, 512, 1024]
    calibration_errors = {name: [[] for _ in range(4)] for name in trainer_names}

    for budget in simulation_budgets:
        trainers, names = get_trainers(simulation_budget=budget)
        config_outputs = trainers[0].configurator(generative_model(5000))
        prior_samples = config_outputs["posterior_inputs"]["parameters"]

        for i, _ in enumerate(trainer_names):
            posterior_samples = trainers[i].amortizer.sample_parameters(
                config_outputs, n_samples=1000
            )
            calibration_error = posterior_calibration_error(
                posterior_samples, prior_samples
            )

            for j, error in enumerate(calibration_error):
                calibration_errors[trainer_names[i]][j].append(error)

    with open(COMPUTATIONS_DIR / "hes1_calibration_errors.pkl", "wb") as f:
        pickle.dump(calibration_errors, f)

    return calibration_errors


def calibration_error_plot():
    cmap = plt.get_cmap("viridis", 3)
    fig, axarr = plt.subplots(1, 4, figsize=(20, 6))

    with open(COMPUTATIONS_DIR / "hes1_calibration_errors.pkl", "rb") as f:
        calibration_errors = pickle.load(f)

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
    param_names = [r"$p_0$", r"$h$", r"$k_1$", r"$\nu$"]
    linestyles = ["solid", "dotted", "dashed"]

    for i, ax in enumerate(axarr):
        for j, estimator in enumerate(estimators):
            label = estimators[estimator]["label"]
            color = estimators[estimator]["color"]

            ax.plot(
                range(len(simulation_budgets)),
                calibration_errors[estimator][i],
                color=color,
                linestyle=linestyles[j],
                linewidth=3.0
            )
            ax.plot(
                range(len(simulation_budgets)),
                calibration_errors[estimator][i],
                marker="o",
                color=color,
                linestyle=linestyles[j],
                markersize=10,
                label=label if i == 3 else None,
            )

        print(i)
        ax.set_title(param_names[i], fontsize=40)
        ax.set_ylim(0, None)
        ax.set_xticks(range(len(simulation_budgets)))
        ax.tick_params(axis="both", which="major", labelsize=26)
        ax.set_xticklabels(
            [rf"$2^{{{int(l)}}}$" for l in np.log2(np.array(simulation_budgets))]
        )

    sns.despine()
    fig.supxlabel("Simulation budget", fontsize=36, y=0.15)
    fig.supylabel("Calibration error", fontsize=36)
    fig.legend(fontsize=28, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3)

    fig.tight_layout()
    fig.subplots_adjust(left=0.1, bottom=0.35)

    fig.savefig(PLOT_DIR / "hes_1_simulation_budget_calibration.pdf")

    return calibration_errors


if __name__ == "__main__":
    calibration_error_savefile = COMPUTATIONS_DIR / "hes1_calibration_errors.pkl"

    if not calibration_error_savefile.exists():
        calibration_error_eval()

    calibration_error_plot()
