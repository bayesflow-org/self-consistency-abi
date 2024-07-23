import math
import tensorflow_probability as tfp
import bayesflow as bf
import tensorflow as tf
import numpy as np
import itertools

from sc_abi.sc_simulation import PriorLogProb
import tasks.location_finding_utils as lf


## FLATTENED VERSION ##########################################


def get_prior_dist(K: int):
    means = tf.zeros((K * 2))
    st_devs = tf.ones((K * 2))
    return tfp.distributions.MultivariateNormalDiag(loc=means, scale_diag=st_devs)


def get_inference_network(K: int):
    return bf.networks.InvertibleNetwork(
        num_params=2 * K,  # number of sources times 2D
        num_coupling_layers=6,
        coupling_design="spline",
        permutation="fixed",
        # coupling_settings={"kernel_regularizer": tf.keras.regularizers.l2(1e-4)},
    )


def get_summary_network(K: int):
    # scale summary dimension with the number of parameters
    # return bf.networks.DeepSet(summary_dim=4 * K)
    return bf.networks.SetTransformer(
        input_dim=3,
        # attention_settings=dict(num_heads=16, key_dim=32),
        use_layer_norm=True,
        # num_attention_blocks=6,
        summary_dim=8 * K,
    )


# latent distribution: Student-t with 100 DoF (only to slightly influence tail behavior)
class MultimodalLatent(tfp.distributions.MixtureSameFamily):
    def __init__(self, K: int):
        mixture_weights_dist = tfp.distributions.Categorical(probs=[1 / K] * K)
        t = np.linspace(0, 2 * np.pi, K + 1)[:K]
        locations = [
            np.array([1.5 * np.cos(t_k), 1.5 * np.sin(t_k)], dtype=np.float32)
            for t_k in t
        ]

        components_dist = tfp.distributions.MultivariateNormalDiag(
            loc=tf.stack(locations, axis=0), scale_diag=[0.2] * 2
        )

        super().__init__(
            mixture_distribution=mixture_weights_dist,
            components_distribution=components_dist,
        )


# foo = MultimodalLatent(5)
# samples = foo.sample(100)
# plt.scatter(samples[:, 0], samples[:, 1])

# def sample(self, *args, **kwargs):
#     # overwrite sample method?
#     samples = super().sample(*args, **kwargs)
#     return samples.reshape(-1, 2 * K)


class MultimodalLatentPerms(tfp.distributions.MixtureSameFamily):
    def __init__(self, K: int):
        num_perms = math.factorial(K)
        mixture_weights_dist = tfp.distributions.Categorical(
            probs=[1 / num_perms] * num_perms
        )
        base_matrix = np.concatenate(
            [(-1) ** i * np.eye(K, dtype=np.float32) for i in range(K // 2 + 1)],
            axis=0,
        )[:K]
        # generate all matrices with permuted rows form base_matrix
        locations = [
            base_matrix[p, :].reshape(-1) for p in itertools.permutations(range(K))
        ]
        components_dist = tfp.distributions.MultivariateNormalDiag(
            loc=tf.stack(locations, axis=0), scale_diag=[0.2] * 2 * K
        )

        super().__init__(
            mixture_distribution=mixture_weights_dist,
            components_distribution=components_dist,
        )


def get_latent_dist(K: int, df: int):
    return tfp.distributions.MultivariateStudentTLinearOperator(
        df=df,
        loc=[0.0] * 2 * K,
        scale=tf.linalg.LinearOperatorDiag([1.0] * 2 * K),
    )
    # K independent MulimodalLatent distributions
    # return MultimodalLatent()
    # return MultimodalLatentPerms()


def get_amortizer_arguments(K: int, latent_df: int = 50):
    return {
        "inference_net": get_inference_network(K=K),
        "summary_net": get_summary_network(K=K),
        "latent_dist": get_latent_dist(K=K, df=latent_df),
    }


class LocationFinding:
    def __init__(
        self,
        K: int = 1,
        observations_sd: float = 0.5,
        sample_measurement_points: bool = False,
        physical_dim: int = 2,
    ):
        if sample_measurement_points:
            self.p = physical_dim
            x = np.random.uniform(-3.5, 3.5, (30, physical_dim))
            self.x = tf.convert_to_tensor(x, dtype=tf.float32)
            self.M = self.x.shape[0]  # number of measurement points
        else:
            self.p = 2
            # x contains 30 mreasurement points
            self.x = tf.convert_to_tensor(lf.MEASUREMENT_POINTS, dtype=tf.float32)
            self.M = self.x.shape[0]

        self.K = K  # number of sources
        # make this a tensor of shape [(B), M, 1]
        # expand observations_sd to shape [(B), M, 1]
        observations_sd = tf.ones((self.M, 1)) * observations_sd
        self.observations_sd = observations_sd
        self.dist = tfp.distributions.Normal

    def _build_dist(self, theta):
        # DRI: fixed reshaping
        # print("build_dist theta-->", theta.shape)
        new_shape = [*theta.shape[:-1], self.K, 2]  # [(B), K, 2]
        theta = tf.reshape(theta, new_shape)
        # print("build_dist theta reshaped-->", theta.shape)
        observations_mean = lf.get_mean_tf(theta, self.x)  # [(B), M]
        # print("observations_mean.shape", observations_mean.shape)
        dist = self.dist(loc=observations_mean, scale=self.observations_sd)
        # mark event and batch shapes
        dist = tfp.distributions.Independent(
            dist, reinterpreted_batch_ndims=1, name="observations"
        )
        return dist

    def __call__(self, theta):
        dist = self._build_dist(theta)

        y = dist.sample()
        # concat y and self.x
        # expand x to have the same batch size as y
        x = tf.expand_dims(self.x, 0)
        x = tf.tile(x, [tf.shape(y)[0], 1, 1])  # type: ignore
        output = tf.concat([y, x], axis=-1)
        return output  # [(B), M, 3]

    def log_prob(self, theta, y):
        # returns the logprobs of the dataset
        dist = self._build_dist(theta)
        # sum over data
        return tf.reduce_sum(dist.log_prob(y), axis=-1)
