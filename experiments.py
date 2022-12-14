import sys

from experiments.decision_tree import execute as decision_tree_experiment
from experiments.scores_all_regs import execute as all_regressors_experiment
from experiments.embedded_regs import execute as embedded_regressors_experiment
from experiments.evaluate_classifiers import execute as evaluate_classifiers_experiment


def main():

    if len(sys.argv) < 2:
        print("Must be called as: python experiments.py --<experiment_name>")
    if sys.argv[1] == "--decision_tree":
        decision_tree_experiment()
    elif sys.argv[1] == "--all_regressors":
        all_regressors_experiment()
    elif sys.argv[1] == "--embedded_regressors":
        embedded_regressors_experiment()
    elif sys.argv[1] == "--evaluate_classifiers":
        evaluate_classifiers_experiment()
    else:
        print("Unknown experiment name")


if __name__ == "__main__":
    main()
