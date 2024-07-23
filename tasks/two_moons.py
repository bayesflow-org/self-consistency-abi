import bayesflow as bf
import numpy as np
import tensorflow as tf

from sc_abi.sc_simulation import PriorLogProb

prior_dist = tfp.distributions.Independent(
    tfp.distributions.Uniform(low=[-2.0, -2.0], high=[2.0, 2.0]),
    reinterpreted_batch_ndims=1,
)
prior = PriorLogProb(prior_dist)

generative_model = bf.simulation.GenerativeModel(
    prior=prior,
    simulator=two_moons.simulator,
    prior_is_batched=True,
    simulator_is_batched=False,
)


def get_amortizer_arguments():
    return {
        "amortized_posterior": get_amortized_posterior(),
        "amortized_likelihood": get_amortized_likelihood(),
    }


def get_amortized_posterior():
    return bf.amortizers.AmortizedPosterior(
        bf.networks.InvertibleNetwork(
            num_params=2,
            # num_coupling_layers=5,
            # coupling_design="interleaved",
            # permutation="learnable"
            num_coupling_layers=6,
            coupling_design="spline",
            coupling_settings={
                "dense_args": dict(units=128),
                "kernel_regularizer": tf.keras.regularizers.l2(1e-4),
            },
        ),
        latent_dist=get_latent_dist(),
    )


def get_amortized_likelihood():
    return bf.amortizers.AmortizedLikelihood(
        bf.networks.InvertibleNetwork(
            num_params=2,
            num_coupling_layers=6,
            coupling_design="spline",
            coupling_settings={
                "dense_args": dict(units=128),
                "kernel_regularizer": tf.keras.regularizers.l2(1e-4),
            },
        ),
        latent_dist=get_latent_dist(),
    )


def get_latent_dist():
    return tfp.distributions.MultivariateStudentTLinearOperator(
        df=50,
        loc=[0.0] * 2,
        scale=tf.linalg.LinearOperatorDiag([1.0] * 2),
    )


def simulator_numpy(theta, rng=None):
    # Use default RNG, if None specified
    if rng is None:
        rng = np.random.default_rng()

    # Generate noise
    alpha = rng.uniform(low=-0.5 * np.pi, high=0.5 * np.pi)
    r = rng.normal(loc=0.1, scale=0.01)

    # Forward process
    rhs1 = np.array([r * np.cos(alpha) + 0.25, r * np.sin(alpha)])
    rhs2 = np.array(
        [
            -np.abs(theta[0] + theta[1]) / np.sqrt(2.0),
            (-theta[0] + theta[1]) / np.sqrt(2.0),
        ]
    )

    return rhs1 + rhs2


def analytic_posterior_numpy(x_o, n_samples=1, rng=None):
    ang = -np.pi / 4.0
    c = np.cos(-ang)
    s = np.sin(-ang)

    theta = np.zeros((n_samples, 2), dtype=np.float32)

    for i in range(n_samples):
        p = simulator_numpy(np.zeros(2), rng=rng)
        q = np.zeros(2)

        q[0] = p[0] - x_o[0]
        q[1] = x_o[1] - p[1]

        if np.random.rand() < 0.5:
            q[0] = -q[0]

        theta[i, 0] = c * q[0] - s * q[1]
        theta[i, 1] = s * q[0] + c * q[1]

    return theta.astype(np.float32)
