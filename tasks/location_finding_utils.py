import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Looked at bayesflow.benchmarks.two_moons
K = 2  # Number of sources

# Measurement points where we will observe the signal strength
MEASUREMENT_POINTS = np.array(
    [
        [1.3971596, 1.0928553],
        [-0.5840266, -0.24283463],
        [-0.45586693, 0.17640619],
        [0.19403903, 0.9742167],
        [0.06762153, -0.02957348],
        [1.6097187, -0.6532622],
        [-1.5166168, 0.64095485],
        [1.4650464, 0.28684628],
        [-0.78284544, 0.61133236],
        [0.7968642, 0.79689676],
        [-0.679654, -1.5442262],
        [0.18040109, -1.5498956],
        [1.0577332, -0.1846191],
        [-0.24410132, -0.9719644],
        [-1.0414574, 1.3571163],
        [-1.3331215, -0.9833422],
        [0.23092182, -0.7919835],
        [0.43889588, 0.45813093],
        [0.50383914, 1.6309954],
        [-0.15232024, -0.4068564],
        [0.662421, 0.09915663],
        [0.96514744, -1.387434],
        [0.7765432, -0.7305141],
        [-1.0337799, 0.07204529],
        [0.41157117, -0.3334951],
        [-0.01501311, 0.37219712],
        [-1.5838764, -0.24988894],
        [-0.72755647, -0.70040244],
        [-0.29137558, 1.4146744],
        [-0.30416486, 0.7523038],
    ]
)


def prior(rng=None):
    """Generates a random draw from a Gaussian prior for the locations of K sources."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(size=(K, 2))


def get_mean(theta, measurement_points):
    """
    Computes the mean of the signal strength at the measurement points.

    theta : np.ndarray of shape [(B), K, 2)
        Locations of the K sources.
    measurement_points : np.ndarray of shape (M, 2)
        Locations of the M measurement points.
    """
    max_signal = 1e-4  # added before taking the inverse
    base_signal = 0.1  # added before taking the log

    theta_expanded = theta[..., np.newaxis, :, :]  # [(B), 1, K, 2]
    # print("theta_expanded.shape", theta_expanded.shape)
    measurements = measurement_points[:, np.newaxis, :]  # [M, 1, 2]

    sq_two_norm = np.sum(
        np.square(measurements - theta_expanded), axis=-1
    )  # [(B), M, K]
    # print("sq_two_norm.shape", sq_two_norm.shape)
    sq_two_norm_inverse = np.power(max_signal + sq_two_norm, -1)  # [(B), M, K]
    # sum over the K sources
    mean_y = np.log(base_signal + np.sum(sq_two_norm_inverse, axis=-1))  # [(B), M]
    # print("mean_y.shape", mean_y.shape)

    # unsqueeze at the end
    # mean_y = mean_y[..., np.newaxis]  # [(B), M, 1]
    return mean_y


def get_mean_tf(theta, measurement_points):
    """
    Computes the mean of the signal strength at the measurement points.

    theta : np.ndarray of shape [(B), K, 2)
        Locations of the K sources.
    measurement_points : np.ndarray of shape (M, 2)
        Locations of the M measurement points.
    """
    max_signal = 1e-4  # added before taking the inverse
    base_signal = 0.1  # added before taking the log

    # Expand theta to shape [(B), 1, K, 2]
    theta_expanded = tf.expand_dims(theta, axis=-3)

    # Expand measurement points to shape [M, 1, 2]
    measurements = tf.expand_dims(measurement_points, axis=1)

    # Calculate the squared two-norm
    sq_two_norm = tf.reduce_sum(tf.square(measurements - theta_expanded), axis=-1)

    # Inverse of squared two-norm
    sq_two_norm_inverse = tf.math.reciprocal(max_signal + sq_two_norm)

    # Sum over the K sources and add base_signal -> shape [(B), M]
    mean_y = tf.math.log(base_signal + tf.reduce_sum(sq_two_norm_inverse, axis=-1))

    # Expand mean_y to shape [(B), M, 1]
    mean_y = tf.expand_dims(mean_y, axis=-1)

    return mean_y


def simulator(theta, rng=None, measurement_points=MEASUREMENT_POINTS):
    """
    Simulates the signal strength based on the locations of the sources.

    theta : np.ndarray of shape (K, 2)
        Locations of the K sources.
    rng : np.random.Generator or None
        Random number generator for noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise_sd = 0.5
    mean_y = get_mean(theta, measurement_points)

    # add gsn noise
    return mean_y + rng.normal(scale=noise_sd, size=mean_y.shape)


def plot_sources_and_measurements():
    # cmap_measurements = LinearSegmentedColormap.from_list(
    #     "custom_red_green", ["red", "green"], N=256
    # )
    cmap_sources = LinearSegmentedColormap.from_list(
        "custom_purple", ["#f2f2ff", "#540354"], N=256
    )
    rng = np.random.default_rng(1)
    theta_sample = prior(rng=rng)
    y_samples = simulator(theta_sample, rng=rng)
    print(theta_sample.shape)
    print(y_samples.shape)

    # Creating a grid for the background signal
    min, max = -2.5, 2.5
    xx, yy = np.meshgrid(np.linspace(min, max, 400), np.linspace(min, max, 400))

    # Simulate the signal on the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    signal_grid = get_mean(theta=theta_sample, measurement_points=grid_points).reshape(
        xx.shape
    )
    norm = Normalize(vmin=signal_grid.min(), vmax=signal_grid.max())
    # Plot the signal on the grid
    plt.figure(figsize=(8, 5))
    imshow = plt.imshow(
        signal_grid,
        extent=(min, max, min, max),
        origin="lower",
        cmap=cmap_sources,
        norm=norm,
        alpha=0.7,
        aspect=0.82,
    )
    plt.colorbar(imshow, label="Signal Intensity")
    plt.scatter(
        theta_sample[:, 0],
        theta_sample[:, 1],
        color="black",
        label="Source locations",
        marker="*",
        # make the star marker without fill, just black border
        edgecolors="black",
        facecolors="none",
        alpha=0.4,
        s=50,
    )
    scatter = plt.scatter(
        MEASUREMENT_POINTS[:, 0],
        MEASUREMENT_POINTS[:, 1],
        c=y_samples,
        cmap=cmap_sources,
        norm=norm,
        edgecolors="black",
        label="Measurements",
        alpha=0.7,
    )
    # remove ticks
    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,  # ticks along the top edge are off
        labelbottom=False,
        labelleft=False,
    )
    # plt.title("Source Locations and Measurement Points")
    plt.legend()
    # save the figure in pdf
    plt.savefig("source_locations.pdf", bbox_inches="tight")
    # plt.show()
    plt.close("all")


if __name__ == "__main__":
    plot_sources_and_measurements()
