from pathlib import Path
import numpy as np
import seaborn as sns
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
from bayesflow.computational_utilities import simultaneous_ecdf_bands
from bayesflow.helper_functions import check_posterior_prior_shapes

from scripts.hes1_nple_train import get_trainers
from tasks.hes1 import custom_plot_sbc_ecdf, generative_model

PLOT_DIR = Path(__file__).parents[1] / "plots"


def custom_plot_sbc_ecdf(
        post_samples,
        prior_samples,
        difference=False,
        stacked=False,
        fig_size=None,
        param_names=None,
        label_fontsize=16,
        title_fontsize=18,
        tick_fontsize=16,
        rank_ecdf_color="#a34f4f",
        fill_color="grey",
        **kwargs,
):
    # Sanity checks
    check_posterior_prior_shapes(post_samples, prior_samples)

    # Store reference to number of parameters
    n_params = post_samples.shape[-1]

    # Compute fractional ranks (using broadcasting)
    ranks = (
            np.sum(post_samples < prior_samples[:, np.newaxis, :], axis=1)
            / post_samples.shape[1]
    )

    # Prepare figure
    if stacked:
        n_row, n_col = 1, 1
        f, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        # Determine n_subplots dynamically
        n_row = int(np.ceil(n_params / 6))
        n_col = int(np.ceil(n_params / n_row))

        # Determine fig_size dynamically, if None
        if fig_size is None:
            fig_size = (int(5 * n_col), int(5 * n_row))

        # Initialize figure
        f, ax = plt.subplots(n_row, n_col, figsize=fig_size)
    # Compute uniform ECDF and bands
    alpha, z, L, H = simultaneous_ecdf_bands(
        post_samples.shape[0], **kwargs.pop("ecdf_bands_kwargs", {})
    )

    # Difference, if specified
    if difference:
        L -= z
        H -= z
        ylab = "ECDF difference"
    else:
        ylab = "ECDF"

    # Plot individual ecdf of parameters
    for j in range(ranks.shape[-1]):
        ecdf_single = np.sort(ranks[:, j])
        xx = ecdf_single
        yy = np.arange(1, xx.shape[-1] + 1) / float(xx.shape[-1])

        # Difference, if specified
        if difference:
            yy -= xx

        if stacked:
            if j == 0:
                ax.plot(
                    xx,
                    yy,
                    color=rank_ecdf_color,
                    alpha=0.95,
                    linewidth=4,
                    label="Rank ECDFs",
                )
            else:
                ax.plot(xx, yy, color=rank_ecdf_color, alpha=0.95, linewidth=4)
        else:
            ax.flat[j].plot(
                xx,
                yy,
                color=rank_ecdf_color,
                alpha=0.95,
                linewidth=3,
                label="Rank ECDF",
            )

    # Add simultaneous bounds
    if stacked:
        titles = [None]
        axes = [ax]
    else:
        axes = ax.flat
        if param_names is None:
            titles = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]
        else:
            titles = param_names

    for _ax, title in zip(axes, titles):
        _ax.fill_between(
            z,
            L,
            H,
            color=fill_color,
            alpha=0.2,
            label=rf"{int((1 - alpha) * 100)}$\%$ Confidence Bands",
        )

        # Prettify plot
        sns.despine(ax=_ax)
        _ax.grid(alpha=0.35)
        # _ax.legend(fontsize=legend_fontsize)
        _ax.set_title(title, fontsize=title_fontsize)
        _ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        _ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Only add x-labels to the bottom row
    # if stacked:
    #     bottom_row = [ax]
    # else:
    #     bottom_row = ax if n_row == 1 else ax[-1, :]
    # for _ax in bottom_row:
    #     _ax.set_xlabel("Fractional rank statistic", fontsize=label_fontsize)

    # Only add y-labels to right left-most row
    if n_row == 1:  # if there is only one row, the ax array is 1D
        axes[0].set_ylabel(ylab, fontsize=label_fontsize)
    else:  # if there is more than one row, the ax array is 2D
        for _ax in ax[:, 0]:
            _ax.set_ylabel(ylab, fontsize=label_fontsize)

    # Remove unused axes entirely
    for _ax in axes[n_params:]:
        _ax.remove()
    f.supxlabel("Fractional rank statistic", fontsize=label_fontsize)
    f.tight_layout()
    return f


def sbc_ecdf_plots():
    trainers, names = get_trainers(simulation_budget=512)

    test_sims = trainers[0].configurator(generative_model(500))
    posterior_samples = [
        trainer.amortizer.sample_parameters(test_sims, n_samples=500)
        for trainer in trainers
    ]

    for i, posterior_sample in enumerate(posterior_samples[::-1]):
        if i == 0:
            continue

        print(plt.get_cmap("viridis", 2)(i))
        f = custom_plot_sbc_ecdf(
            posterior_sample,
            test_sims["posterior_inputs"]["parameters"],
            difference=True,
            param_names=[r"$p_0$", r"$h$", r"$k_1$", r"$\nu$"],
            label_fontsize=36,
            title_fontsize=40,
            tick_fontsize=26,
            rank_ecdf_color=plt.get_cmap("viridis", 3)(i),
        )
        f.savefig(PLOT_DIR / f"hes1_calibration_{names[i]}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    sbc_ecdf_plots()
