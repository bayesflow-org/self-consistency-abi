import numpy as np
import tensorflow as tf
from bayesflow.amortizers import AmortizedPosterior, AmortizedPosteriorLikelihood
from bayesflow.default_settings import DEFAULT_KEYS

from sc_abi.sc_schedules import ConstantSchedule


class AmortizedPosteriorSC(AmortizedPosterior):
    def __init__(
        self,
        prior,
        simulator,
        lambda_schedule=ConstantSchedule(1.0),
        n_consistency_samples=10,
        theta_clip_value_min=-float("inf"),
        theta_clip_value_max=float("inf"),
        output_numpy=False,
        mode="sc",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prior = prior
        self.simulator = simulator
        self.step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.lambda_schedule = lambda_schedule
        self.n_consistency_samples = n_consistency_samples
        self.output_numpy = output_numpy
        self.theta_clip_value_min = theta_clip_value_min
        self.theta_clip_value_max = theta_clip_value_max
        self.mode = mode

    def compute_loss(self, input_dict, **kwargs):
        self.step.assign_add(1)
        lamda = self.lambda_schedule(self.step)

        # Get amortizer outputs
        net_out, sum_out = self(input_dict, return_summary=True, **kwargs)
        z, log_det_J = net_out

        # Case summary loss should be computed
        if self.summary_loss is not None:
            sum_loss = self.summary_loss(sum_out)
        # Case no summary loss, simply add 0 for convenience
        else:
            sum_loss = 0.0

        # Case dynamic latent space - function of summary conditions
        if self.latent_is_dynamic:
            logpdf = self.latent_dist(sum_out).log_prob(z)
        # Case _static latent space
        else:
            logpdf = self.latent_dist.log_prob(z)

        # Compute and return total posterior loss
        posterior_loss = tf.reduce_mean(-logpdf - log_det_J) + sum_loss

        # SELF CONSISTENCY LOSS

        if input_dict.get(DEFAULT_KEYS["summary_conditions"]) is not None:
            x = input_dict.get(DEFAULT_KEYS["summary_conditions"])
        else:
            x = input_dict.get(DEFAULT_KEYS["direct_conditions"])

        condition = x if self.summary_net is None else sum_out

        theta_true = input_dict[DEFAULT_KEYS["parameters"]]
        batch_size, *param_dim = theta_true.shape
        n_consistency_samples = self.n_consistency_samples - 1

        if tf.greater(lamda, tf.constant(0.0)):
            z = self.latent_dist.sample(
                (batch_size, n_consistency_samples)
            )  # batch_size, n_consistency_samples, param_dim
            theta = tf.stop_gradient(
                self.inference_net.inverse(z, condition, training=False)
            )

            theta = tf.concat(
                [
                    theta,
                    tf.convert_to_tensor(theta_true, dtype=tf.float32)[
                        :, tf.newaxis, :
                    ],
                ],
                axis=1,
            )
            theta = tf.clip_by_value(
                theta,
                clip_value_min=self.theta_clip_value_min,
                clip_value_max=self.theta_clip_value_max,
            )

            log_prior = self.prior.log_prob(theta)

            if self.output_numpy:
                theta = tf.make_ndarray(theta)
                log_lik = np.empty((batch_size, n_consistency_samples + 1))
                for i in range(n_consistency_samples + 1):
                    th = theta[:, i, :]
                    log_lik[:, i] = self.simulator.log_prob(th, x)
                log_lik = tf.convert_to_tensor(log_lik, dtype=tf.float32)
            else:
                log_lik = tf.vectorized_map(
                    lambda th: self.simulator.log_prob(th, x),
                    tf.transpose(theta, [1, 0, 2]),
                )
                log_lik = tf.transpose(log_lik, [1, 0])

            updated_dict = input_dict.copy()
            updated_dict[DEFAULT_KEYS["parameters"]] = theta
            log_post = self.log_posterior(updated_dict, to_numpy=False)

            log_ml = log_prior + log_lik - log_post

            if self.mode == "sc":
                log_ml_var = tf.math.reduce_variance(log_ml, axis=1)
                sc_loss = tf.math.reduce_mean(log_ml_var)
            else:
                raise ValueError(f"Mode {self.mode} not recognized.")
        else:
            sc_loss = tf.constant(0.0)

        return {"Post.Loss": posterior_loss, "SC.Loss": lamda * sc_loss}


class AmortizedPosteriorLikelihoodSC(AmortizedPosteriorLikelihood):
    def __init__(
        self,
        prior,
        lambda_schedule=ConstantSchedule(1.0),
        n_consistency_samples=10,
        theta_clip_value_min=-float("inf"),
        theta_clip_value_max=float("inf"),
        output_numpy=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prior = prior
        self.step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.lambda_schedule = lambda_schedule
        self.n_consistency_samples = n_consistency_samples
        self.output_numpy = output_numpy
        self.theta_clip_value_min = theta_clip_value_min
        self.theta_clip_value_max = theta_clip_value_max

    def compute_loss(self, input_dict, **kwargs):
        self.step.assign_add(1)
        lamda = self.lambda_schedule(self.step)

        # POSTERIOR LOSS ####
        # Get amortizer outputs
        posterior_input_dict = input_dict[DEFAULT_KEYS["posterior_inputs"]]
        net_out, sum_out = self.amortized_posterior(
            posterior_input_dict, return_summary=True, **kwargs
        )
        z, log_det_J = net_out

        # Case summary loss should be computed
        if self.amortized_posterior.summary_loss is not None:
            sum_loss = self.amortized_posterior.summary_loss(sum_out)
        # Case no summary loss, simply add 0 for convenience
        else:
            sum_loss = 0.0

        # Case dynamic latent space - function of summary conditions
        if self.amortized_posterior.latent_is_dynamic:
            logpdf = self.amortized_posterior.latent_dist(sum_out).log_prob(z)
        # Case _static latent space
        else:
            logpdf = self.amortized_posterior.latent_dist.log_prob(z)

        # Compute and return total posterior loss
        posterior_loss = tf.reduce_mean(-logpdf - log_det_J) + sum_loss

        # LIKELIHOOD LOSS ####

        likelihood_input_dict = input_dict[DEFAULT_KEYS["likelihood_inputs"]]
        z_lik, log_det_J_lik = self.amortized_likelihood(
            likelihood_input_dict, **kwargs
        )
        likelihood_loss = tf.reduce_mean(
            -self.amortized_likelihood.latent_dist.log_prob(z_lik) - log_det_J_lik
        )

        # SELF CONSISTENCY LOSS ####
        if tf.greater(lamda, 0.0):
            if posterior_input_dict.get(DEFAULT_KEYS["summary_conditions"]) is not None:
                x = posterior_input_dict.get(DEFAULT_KEYS["summary_conditions"])
            else:
                x = posterior_input_dict.get(DEFAULT_KEYS["direct_conditions"])

            condition = x if self.amortized_posterior.summary_net is None else sum_out

            theta_true = posterior_input_dict[DEFAULT_KEYS["parameters"]]
            batch_size, param_dim = tf.shape(theta_true)[0], tf.shape(theta_true)[1]
            n_consistency_samples = self.n_consistency_samples - 1

            x = likelihood_input_dict[DEFAULT_KEYS["observables"]]

            z = self.amortized_posterior.latent_dist.sample(
                (batch_size, n_consistency_samples)
            )  # batch_size, n_con, param_dim
            theta = tf.stop_gradient(
                self.amortized_posterior.inference_net.inverse(
                    z, condition, training=False
                )
            )
            theta = tf.concat(
                [
                    theta,
                    tf.convert_to_tensor(theta_true, dtype=tf.float32)[
                        :, tf.newaxis, :
                    ],
                ],
                axis=1,
            )

            theta = tf.clip_by_value(
                theta,
                clip_value_min=self.theta_clip_value_min,
                clip_value_max=self.theta_clip_value_max,
            )

            log_prior = self.prior.log_prob(theta)

            theta_transposed = tf.transpose(theta, [1, 0, 2])
            log_lik = tf.map_fn(
                lambda th: self.log_likelihood(
                    {"conditions": th, "observables": x}, to_numpy=False
                ),
                theta_transposed,
            )

            if log_lik.ndim == 2:
                log_lik = tf.transpose(log_lik, [1, 0])

            elif log_lik.ndim == 3:
                log_lik = tf.transpose(log_lik, [1, 0, 2])
                log_lik = tf.reduce_sum(log_lik, axis=-1)

            # log_lik = self.log_likelihood({'conditions': theta, 'observables': x}, to_numpy=False)

            updated_dict = posterior_input_dict.copy()
            updated_dict[DEFAULT_KEYS["parameters"]] = theta
            log_post = self.log_posterior(updated_dict, to_numpy=False)

            log_ml = log_prior + log_lik - log_post
            log_ml_var = tf.math.reduce_variance(log_ml, axis=1)

            sc_loss = tf.math.reduce_mean(log_ml_var)
        else:
            sc_loss = tf.constant(0.0)

        return {
            "Post.Loss": posterior_loss,
            "Lik.Loss": likelihood_loss,
            "SC.Loss": sc_loss * lamda,
        }


class AmortizedPosteriorLikelihoodSCSingleObs(AmortizedPosteriorLikelihoodSC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, input_dict, **kwargs):
        self.step.assign_add(1)
        lamda = self.lambda_schedule(self.step)

        # POSTERIOR LOSS ####
        # Get amortizer outputs
        posterior_input_dict = input_dict[DEFAULT_KEYS["posterior_inputs"]]
        net_out, sum_out = self.amortized_posterior(
            posterior_input_dict, return_summary=True, **kwargs
        )
        z, log_det_J = net_out

        # Case summary loss should be computed
        if self.amortized_posterior.summary_loss is not None:
            sum_loss = self.amortized_posterior.summary_loss(sum_out)
        # Case no summary loss, simply add 0 for convenience
        else:
            sum_loss = 0.0

        # Case dynamic latent space - function of summary conditions
        if self.amortized_posterior.latent_is_dynamic:
            logpdf = self.amortized_posterior.latent_dist(sum_out).log_prob(z)
        # Case _static latent space
        else:
            logpdf = self.amortized_posterior.latent_dist.log_prob(z)

        # Compute and return total posterior loss
        posterior_loss = tf.reduce_mean(-logpdf - log_det_J) + sum_loss

        # LIKELIHOOD LOSS ####

        likelihood_input_dict = input_dict[DEFAULT_KEYS["likelihood_inputs"]]
        z_lik, log_det_J_lik = self.amortized_likelihood(
            likelihood_input_dict, **kwargs
        )
        likelihood_loss = tf.reduce_mean(
            -self.amortized_likelihood.latent_dist.log_prob(z_lik) - log_det_J_lik
        )

        # SELF CONSISTENCY LOSS ####
        if posterior_input_dict.get(DEFAULT_KEYS["summary_conditions"]) is not None:
            x = posterior_input_dict.get(DEFAULT_KEYS["summary_conditions"])
        else:
            x = posterior_input_dict.get(DEFAULT_KEYS["direct_conditions"])

        condition = x if self.amortized_posterior.summary_net is None else sum_out

        theta_true = posterior_input_dict[DEFAULT_KEYS["parameters"]]
        batch_size, param_dim = tf.shape(theta_true)[0], tf.shape(theta_true)[1]
        n_consistency_samples = self.n_consistency_samples - 1
        # theta sampling scheme:
        # - if posterior is probably sufficiently good: sample theta ~ posterior
        # - else: sample theta ~ prior
        if tf.greater(lamda, 0.0):
            x = likelihood_input_dict[DEFAULT_KEYS["observables"]]

            z = self.amortized_posterior.latent_dist.sample(
                (batch_size, n_consistency_samples)
            )  # batch_size, n_con, param_dim
            theta = tf.stop_gradient(
                self.amortized_posterior.inference_net.inverse(
                    z, condition, training=False
                )
            )
            theta = tf.concat(
                [
                    theta,
                    tf.convert_to_tensor(theta_true, dtype=tf.float32)[
                        :, tf.newaxis, :
                    ],
                ],
                axis=1,
            )

            log_prior = self.prior.log_prob(theta)

            updated_lik_dict = {
                DEFAULT_KEYS["conditions"]: theta,
                DEFAULT_KEYS["observables"]: tf.tile(
                    x[:, tf.newaxis, :], [1, self.n_consistency_samples, 1]
                ),
            }

            log_lik = self.log_likelihood(updated_lik_dict, to_numpy=False)

            updated_dict = posterior_input_dict.copy()
            updated_dict[DEFAULT_KEYS["parameters"]] = theta
            log_post = self.log_posterior(updated_dict, to_numpy=False)

            log_ml = log_prior + log_lik - log_post
            log_ml_var = tf.math.reduce_variance(log_ml, axis=1)

            sc_loss = tf.math.reduce_mean(log_ml_var)
        else:
            sc_loss = tf.constant(0.0)

        return {
            "Post.Loss": posterior_loss,
            "Lik.Loss": likelihood_loss,
            "SC.Loss": sc_loss * lamda,
        }
