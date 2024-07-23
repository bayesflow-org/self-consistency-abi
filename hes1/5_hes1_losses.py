from pathlib import Path
import bayesflow as bf
from scripts.hes1_nple_train import get_trainers

PLOT_DIR = Path(__file__).parents[1] / "plots"


def plot_losses():
    trainers, names = get_trainers()

    for i, trainer in enumerate(trainers):
        tbl = trainer.loss_history.get_plottable()
        f = bf.diagnostics.plot_losses(tbl["train_losses"], tbl["val_losses"])
        f.savefig(PLOT_DIR / f"hes1_{names[i]}_losses.pdf")


if __name__ == "__main__":
    plot_losses()
