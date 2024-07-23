import tensorflow as tf


class PriorLogProb:
    def __init__(self, dist):
        self.dist = dist

    def __call__(self, batch_size=None):
        theta = self.dist.sample([batch_size]) if batch_size else self.dist.sample()
        return theta

    def log_prob(self, theta):
        logprob = self.dist.log_prob(theta)
        tf.where(tf.math.is_nan(logprob), -float("inf"), logprob)
        return logprob


class SimulatorLogProb:
    def __init__(self, dist, n_obs=10):
        self.dist = dist
        self.n_obs = n_obs

    def __call__(self, theta, n_obs=None):
        if n_obs is None:
            n_obs = self.n_obs
        x = self.dist(theta).sample([n_obs])
        x = tf.transpose(x, perm=[1, 0, 2])
        return x

    def log_prob(self, theta, x):
        x = tf.transpose(x, perm=[1, 0, 2])
        log_ml = self.dist(theta).log_prob(x)
        tf.where(tf.math.is_nan(log_ml), -float("inf"), log_ml)
        log_ml = tf.transpose(log_ml, [1, 0])
        return tf.reduce_sum(log_ml, axis=1)