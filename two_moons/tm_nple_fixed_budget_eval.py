from pathlib import Path

import bayesflow as bf
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from sc_abi.sc_amortizers import AmortizedPosteriorLikelihoodSC
from sc_abi.sc_schedules import ZeroOneSchedule
from tasks.two_moons import generative_model, get_amortizer_arguments, prior

tf.random.set_seed(1)

PLOT_DIR = Path("plots")
CHECKPOINT_DIR = Path("checkpoints/two_moons")

# standard trainer (no self-consistency)

trainer_baseline = bf.trainers.Trainer(
    amortizer=bf.amortizers.AmortizedPosteriorLikelihood(**get_amortizer_arguments()),
    generative_model=generative_model,
    default_lr=0.0005,
    memory=False,
    checkpoint_path=CHECKPOINT_DIR / "npe",
    configurator=bf.benchmarks.Benchmark("two_moons", "joint").configurator,
    max_to_keep=1,
)

trainer_sc = bf.trainers.Trainer(
    amortizer=AmortizedPosteriorLikelihoodSC(
        **get_amortizer_arguments(),
        prior=prior,
        n_consistency_samples=10,
        lambda_schedule=ZeroOneSchedule(threshold_step=128 * 20),
        theta_clip_value_min=-2.0,
        theta_clip_value_max=2.0,
    ),
    generative_model=generative_model,
    default_lr=5e-4,
    memory=False,
    checkpoint_path=CHECKPOINT_DIR / "sc",
    configurator=bf.benchmarks.Benchmark("two_moons", "joint").configurator,
    max_to_keep=1,
)

trainers = [trainer_baseline, trainer_sc]
names = ["NPE", "SC-NPE"]

scatter_kws = {
    "color": "white",
    "alpha": 0.20,
    "rasterized": True,
    "s": 0.7,
    "marker": "D",
}

fig, axes = plt.subplots(1, 2, figsize=(4, 2))
for i, (trainer, name) in enumerate(zip(trainers, names)):
    h = trainer.loss_history.get_plottable()
    _ = bf.diagnostics.plot_losses(h["train_losses"], h["val_losses"])

    samples = trainer.amortizer.sample_parameters(
        {"direct_conditions": np.zeros((1, 2), dtype=np.float32)}, 1000
    )
    axes[i].scatter(samples[:, 0], samples[:, 1], **scatter_kws)
    axes[i].set_title(name)

for ax in axes.flatten():
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.grid(False)
    ax.set_facecolor((0 / 255, 32 / 255, 64 / 255, 1.0))
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.spines["bottom"].set_alpha(0.0)
    ax.spines["top"].set_alpha(0.0)
    ax.spines["right"].set_alpha(0.0)
    ax.spines["left"].set_alpha(0.0)
    ax.set_aspect("equal")

fig.tight_layout()

plt.show()
