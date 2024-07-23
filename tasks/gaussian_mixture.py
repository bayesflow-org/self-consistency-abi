import logging

import bayesflow as bf
import tensorflow as tf
import tensorflow_probability as tfp

from sc_abi.sc_simulation import PriorLogProb, SimulatorLogProb


class GMM(tfp.distributions.MixtureSameFamily):
    def __init__(self, theta):
        logging.getLogger().setLevel(logging.ERROR)
        mixture_weights_dist = tfp.distributions.Categorical(probs=[0.5, 0.5])
        components_dist = tfp.distributions.MultivariateNormalDiag(
            loc=tf.stack([theta, -1.0 * theta], axis=1), scale_diag=[[0.5, 0.5]]
        )

        super().__init__(
            mixture_distribution=mixture_weights_dist,
            components_distribution=components_dist,
        )


prior_dist = tfp.distributions.MultivariateNormalDiag([0.0, 0.0])
prior = PriorLogProb(prior_dist)
simulator = SimulatorLogProb(GMM, n_obs=10)

generative_model = bf.simulation.GenerativeModel(
    prior=prior, simulator=simulator, prior_is_batched=True, simulator_is_batched=True
)


# equivalent neural networks for all architectures


def get_inference_network():
    return bf.networks.InvertibleNetwork(
        num_params=2,
        num_coupling_layers=4,
        coupling_design="spline",
        permutation="learnable",
    )


def get_summary_network():
    return bf.networks.DeepSet(summary_dim=4)


# latent distribution: Student-t with 100 DoF (only to slightly influence tail behavior)
def get_latent_dist():
    return tfp.distributions.MultivariateStudentTLinearOperator(
        df=100,
        loc=[0.0] * 2,
        scale=tf.linalg.LinearOperatorDiag([1.0] * 2),
    )


def get_amortizer_arguments():
    return {
        "inference_net": get_inference_network(),
        "summary_net": get_summary_network(),
        "latent_dist": get_latent_dist(),
    }
