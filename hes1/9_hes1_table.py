import pickle
from pathlib import Path

import numpy as np

COMPUTATIONS_DIR = Path(__file__).parents[1] / "computations"


def hes1_table():
    with open(COMPUTATIONS_DIR / "hes1_eval_dict.pkl", "rb") as f:
        eval_dict = pickle.load(f)

    simulation_budgets = [64]
    run_ids = [1]
    estimators = [
        "baseline",
        "sc",
        "sc_bad_likelihood_net",
    ]

    for _, estimator in enumerate(estimators):
        mmds = eval_dict[estimator][simulation_budgets[0]][run_ids[0]]["mmd"]
        print(estimator)
        print(f"{np.mean(mmds):.3f}")
        print(f"{np.std(mmds):.3f}")
        print("_________")

    return eval_dict


if __name__ == "__main__":
    hes1_table()
