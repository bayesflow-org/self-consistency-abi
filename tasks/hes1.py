import bayesflow as bf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from bayesflow.computational_utilities import simultaneous_ecdf_bands
from bayesflow.helper_functions import check_posterior_prior_shapes
from cmdstanpy import CmdStanModel
from scipy.integrate import solve_ivp
from tqdm import tqdm


class Hes1Dataset:
    data: pd.Series

    def __init__(self):
        # mRNA concentration in fold changes compared to t0
        mRNA = [1.20, 5.90, 4.58, 2.64, 5.38, 6.42, 5.60, 4.48]

        self.data = pd.Series(mRNA, index=range(30, 270, 30))


class Hes1Prior:
    def __init__(self):
        self.dist = tfp.distributions.TransformedDistribution(
            distribution=tfp.distributions.Independent(
                tfp.distributions.Gamma([2.0, 10.0, 2.0, 2.0], [1.0, 1.0, 50.0, 50.0]),
                reinterpreted_batch_ndims=1,
            ),
            bijector=tfp.bijectors.Log(),
        )

    def __call__(self, batch_size=1):
        return self.dist.sample([batch_size])

    def log_prob(self, theta):
        log_prob = self.dist.log_prob(theta)
        log_prob = tf.where(tf.math.is_nan(log_prob), -float("inf"), log_prob)

        return log_prob


def _derivatives(t, y, p0, h, k1, nu):
    m, p1, p2 = y
    d_m = -0.03 * m + 1 / (1 + (p2 / p0) ** h)
    d_p1 = -0.03 * p1 + nu * m - k1 * p1
    d_p2 = -0.03 * p2 + k1 * p1

    return d_m, d_p1, d_p2


def hes1_simulator(params, n_obs=8):
    params = np.exp(params)
    t_span = (0, 270)
    t_eval = [30, 60, 90, 120, 150, 180, 210, 240]

    p0, h, k1, nu = params
    y0 = [2.0, 5.0, 3.0]
    sol = solve_ivp(
        _derivatives, t_span, y0, t_eval=t_eval, args=[p0, h, k1, nu], rtol=1e-7
    )
    y = np.random.normal(loc=sol.y[0, :], scale=1.0, size=(n_obs,)).astype(np.float32)

    return tf.convert_to_tensor(y, dtype=tf.float32)


prior = Hes1Prior()
simulator = bf.simulation.Simulator(simulator_fun=hes1_simulator)

generative_model = bf.simulation.GenerativeModel(
    prior=prior, simulator=simulator, prior_is_batched=True
)


def get_amortizer_arguments():
    return {
        "amortized_posterior": get_amortized_posterior(),
        "amortized_likelihood": get_amortized_likelihood(),
    }


def get_amortized_posterior():
    return bf.amortizers.AmortizedPosterior(
        inference_net=bf.networks.InvertibleNetwork(
            num_params=4,
            num_coupling_layers=4,
            coupling_settings={
                "spec_norm": True,
            },
            coupling_design="spline",
        ),
        latent_dist=get_latent_dist(4),
    )


def get_amortized_likelihood():
    return bf.amortizers.AmortizedLikelihood(
        surrogate_net=bf.networks.InvertibleNetwork(
            num_params=8,
            num_coupling_layers=4,
            coupling_settings={
                "spec_norm": True,
            },
            coupling_design="spline",
        ),
        latent_dist=get_latent_dist(8),
    )


def get_latent_dist(num_params):
    return tfp.distributions.MultivariateStudentTLinearOperator(
        df=50,
        loc=[0.0] * num_params,
        scale=tf.linalg.LinearOperatorDiag([1.0] * num_params),
    )


def configurator(forward_dict):
    input_dict = {}
    input_dict["posterior_inputs"] = {
        "parameters": forward_dict["prior_draws"],
        "direct_conditions": forward_dict["sim_data"],
    }

    input_dict["likelihood_inputs"] = {
        "observables": forward_dict["sim_data"],
        "conditions": forward_dict["prior_draws"],
    }

    return input_dict


def custom_plot_sbc_ecdf(
    post_samples,
    prior_samples,
    difference=False,
    stacked=False,
    fig_size=None,
    param_names=None,
    label_fontsize=16,
    legend_fontsize=14,
    title_fontsize=18,
    tick_fontsize=12,
    rank_ecdf_color="#a34f4f",
    fill_color="grey",
    **kwargs,
):
    """Creates the empirical CDFs for each marginal rank distribution and plots it against
    a uniform ECDF. ECDF simultaneous bands are drawn using simulations from the uniform,
    as proposed by [1].

    For models with many parameters, use `stacked=True` to obtain an idea of the overall calibration
    of a posterior approximator.

    [1] Säilynoja, T., Bürkner, P. C., & Vehtari, A. (2022). Graphical test for discrete uniformity and
    its applications in goodness-of-fit evaluation and multiple sample comparison. Statistics and Computing,
    32(2), 1-21. https://arxiv.org/abs/2103.10522

    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws obtained for generating n_data_sets
    difference        : bool, optional, default: False
        If `True`, plots the ECDF difference. Enables a more dynamic visualization range.
    stacked           : bool, optional, default: False
        If `True`, all ECDFs will be plotted on the same plot. If `False`, each ECDF will
        have its own subplot, similar to the behavior of `plot_sbc_histograms`.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None. Only relevant if `stacked=False`.
    fig_size          : tuple or None, optional, default: None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_fontsize    : int, optional, default: 16
        The font size of the y-label and y-label texts
    legend_fontsize   : int, optional, default: 14
        The font size of the legend text
    title_fontsize    : int, optional, default: 16
        The font size of the title text. Only relevant if `stacked=False`
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    rank_ecdf_color   : str, optional, default: '#a34f4f'
        The color to use for the rank ECDFs
    fill_color        : str, optional, default: 'grey'
        The color of the fill arguments.
    **kwargs          : dict, optional, default: {}
        Keyword arguments can be passed to control the behavior of ECDF simultaneous band computation
        through the ``ecdf_bands_kwargs`` dictionary. See `simultaneous_ecdf_bands` for keyword arguments

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation form the expected shapes of `post_samples` and `prior_samples`.
    """

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
            label=rf"{int((1-alpha) * 100)}$\%$ Confidence Bands",
        )

        # Prettify plot
        sns.despine(ax=_ax)
        _ax.grid(alpha=0.35)
        # _ax.legend(fontsize=legend_fontsize)
        _ax.set_title(title, fontsize=title_fontsize)
        _ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        _ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Only add x-labels to the bottom row
    if stacked:
        bottom_row = [ax]
    else:
        bottom_row = ax if n_row == 1 else ax[-1, :]
    for _ax in bottom_row:
        _ax.set_xlabel("Fractional rank statistic", fontsize=label_fontsize)

    # Only add y-labels to right left-most row
    if n_row == 1:  # if there is only one row, the ax array is 1D
        axes[0].set_ylabel(ylab, fontsize=label_fontsize)
    else:  # if there is more than one row, the ax array is 2D
        for _ax in ax[:, 0]:
            _ax.set_ylabel(ylab, fontsize=label_fontsize)

    # Remove unused axes entirely
    for _ax in axes[n_params:]:
        _ax.remove()

    f.tight_layout()
    return f


def sample_hmc(test_sims, iter_warmup=500, iter_sampling=500, n_chains=4, **kwargs):
    x = test_sims["posterior_inputs"]["direct_conditions"]
    n_sim, n_obs = x.shape
    param_dim = 4

    num_posterior_samples = iter_sampling * n_chains

    posterior_samples = np.zeros((n_sim, num_posterior_samples, param_dim))

    for i in tqdm(range(n_sim), desc="HMC running on data set"):
        stan_data = {
            "N": n_obs,
            "x": [30, 60, 90, 120, 150, 180, 210, 240],
            "y": x[i, :],
        }
        model = CmdStanModel(stan_file="tasks/hes1.stan")
        fit = model.sample(
            data=stan_data,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            chains=n_chains,
            show_progress=False,
            show_console=True,
            **kwargs,
        )
        d = fit.stan_variables()
        posterior_samples[i] = np.stack(
            [d[var_name] for var_name in ["p0", "h", "k1", "nu"]], axis=1
        )

    return np.log(posterior_samples)


def mmd_vectorized(samples_1, samples_2):
    n_sim = samples_1.shape[0]
    assert n_sim == samples_2.shape[0]
    mmd = np.empty([n_sim])
    for i in tqdm(range(n_sim)):
        mmd_value = bf.computational_utilities.maximum_mean_discrepancy(
            samples_1[i].astype(np.float32), samples_2[i].astype(np.float32)
        ).numpy()
        mmd[i] = mmd_value

    return mmd


def ppred(trainer, sim_obs, plot_title="", ppred_color="C0"):
    num_draws = 200

    t_span = (0, 270)
    num_timesteps = 100
    linspace = np.linspace(*t_span, num=num_timesteps)
    posterior_samples = np.exp(trainer.amortizer.sample_parameters(sim_obs, num_draws))

    t_span = (0, 270)
    linspace = np.linspace(*t_span, num=num_timesteps)

    y = np.empty(shape=(num_draws, len(linspace)))

    for i in range(num_draws):
        p0, h, k1, nu = posterior_samples[i,]
        y0 = [2.0, 5.0, 3.0]
        sol = solve_ivp(
            _derivatives, t_span, y0, t_eval=linspace, args=[p0, h, k1, nu], rtol=1e-7
        )
        y[i, :] = sol.y[0, :]  # np.random.normal(sol.y[0, :], 1)

    fig, ax = plt.subplots(1, figsize=(4.8, 2.5))
    plt.grid()
    ax.plot(np.tile(linspace, (num_draws, 1)).T, y.T, color=ppred_color, alpha=0.05)
    ax.scatter(
        Hes1Dataset().data.index,
        Hes1Dataset().data.values,
        color="black",
        zorder=2,
        s=80,
    )
    plt.xticks()
    plt.xlabel(r"$\Delta$t (minutes)", labelpad=16, fontsize=24)
    plt.ylabel("Hes1 mRNA", labelpad=10, fontsize=24)
    plt.ylim(0, 18)
    plt.yticks([0, 5, 10, 15], fontsize=18)

    ax.set_xticks(
        ticks=np.arange(0, 270 + 1, 60),
        labels=np.arange(0, 270 + 1, 60),
        fontsize=18,
        minor=False,
    )
    ax.set_xticks(ticks=np.arange(0, 270 + 1, 30), minor=True)
    ax.grid(axis="x", which="major", alpha=0.6)
    ax.grid(axis="x", which="minor", alpha=0.15)
    plt.xlim(0, 270)
    # plt.title(plot_title, fontsize=16)
    sns.despine()

    return fig
