import tensorflow as tf
import tensorflow_probability as tfp


class ConstantSchedule:
    def __init__(self, value=1.0):
        self.value = value

    def __call__(self, step):
        return self.value


class ZeroOneSchedule:
    def __init__(self, threshold_step, init_step=1):
        self.threshold_step = threshold_step
        self.init_step = init_step

    def __call__(self, step):
        if tf.greater(step, self.threshold_step):
            return 1.0
        else:
            return 0.0


class ZeroLinearOneSchedule:
    def __init__(
        self, threshold1=10.0, threshold2=20.0, init_step=1, lmd: float = 1.0
    ) -> None:
        assert threshold1 < threshold2
        self.threshold1 = tf.cast(threshold1, tf.float32)
        self.threshold2 = tf.cast(threshold2, tf.float32)
        self.init_step = init_step
        self.lmd = tf.cast(lmd, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        if tf.less(step, self.threshold1):
            return tf.cast(0.0, tf.float32)
        elif tf.less_equal(step, self.threshold2):
            return (
                self.lmd
                * (step - self.threshold1)
                / (self.threshold2 - self.threshold1)
            )

        else:
            return self.lmd


class LinearSchedule:
    def __init__(self, max_steps=32 * 100.0, init_step=1):
        self.init_step = init_step
        self.max_steps = max_steps

    def __call__(self, step):
        return tf.cast(step, tf.float32) / self.max_steps


class BetaCDFSchedule:
    def __init__(self, max_steps=32 * 100.0, a=2, b=2, init_step=1):
        self.init_step = init_step
        self.max_steps = max_steps
        self.beta = tfp.distributions.Beta(a, b)

    def __call__(self, step):
        return self.beta.cdf(tf.cast(step, tf.float32) / self.max_steps)
