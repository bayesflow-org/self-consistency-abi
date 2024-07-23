from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scripts.hes1_nple_train import get_trainers
from tasks.hes1 import Hes1Dataset, ppred

PLOT_DIR = Path(__file__).parents[1] / "plots"


def ppc():
    trainers, names = get_trainers()
    sim_obs = {
        "posterior_inputs": {
            "direct_conditions": Hes1Dataset().data.values.astype(np.float32)[
                np.newaxis, :
            ]
        }
    }

    for i, trainer in enumerate(trainers):
        f = ppred(
            trainer,
            sim_obs,
            ppred_color=plt.get_cmap("viridis", 3)(i),
        )
        f.savefig(PLOT_DIR / f"hes1_{names[i]}_ppc.pdf", bbox_inches="tight")


if __name__ == "__main__":
    ppc()
