"""
A script to train a machine learning model on ClimateNet
"""
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


def add_args(parser):
    """
    Add arguments to parser
    """
    parser.add_argument(
        "--train",
        default=None,
        type=str,
        help="Train CSV",
    )
    parser.add_argument(
        "--test",
        default=None,
        type=str,
        help="Test CSV",
    )
    return parser


def main(args):
    # Read data
    if args.train:
        df_tr = pd.read_csv(args.train, index_col=0)
    else:
        print("Train data file missing")
        return
    if args.test:
        df_tt = pd.read_csv(args.test, index_col=0)
    else:
        print("Test data file missing")
        return
    features = [f for f in df_tr.keys() if f not in ["lat", "lon", "time", "LABELS"]]
    x_tr = df_tr[features].values
    x_tt = df_tt[features].values
    y_tr = df_tr.LABELS.values
    y_tt = df_tt.LABELS.values
    print(f"Number of features: {len(features)}")
    print(f"Number of classes: {np.unique(y_tr).shape[0]}")
    print(f"Train size: {x_tr.shape[0]}")
    print(f"Test size: {x_tt.shape[0]}")

    # Normalize
    mean_tr = np.mean(x_tr, axis=0)
    mean_tr = mean_tr[np.newaxis, :]
    std_tr = np.std(x_tr, axis=0)
    std_tr = std_tr[np.newaxis, :]
    x_tr = x_tr - np.tile(mean_tr, (x_tr.shape[0], 1)) / np.tile(
        std_tr, (x_tr.shape[0], 1)
    )
    x_tt = x_tt - np.tile(mean_tr, (x_tt.shape[0], 1)) / np.tile(
        std_tr, (x_tt.shape[0], 1)
    )

    # Train models
    baseline = DummyClassifier(strategy="most_frequent").fit(x_tr, y_tr)
    logreg = LogisticRegression(random_state=0, max_iter=500).fit(x_tr, y_tr)

    # Evaluation
    baseline_acc_tr = baseline.score(x_tr, y_tr)
    baseline_acc_tt = baseline.score(x_tt, y_tt)
    logreg_acc_tt = logreg.score(x_tt, y_tt)
    logreg_acc_tr = logreg.score(x_tr, y_tr)
    logreg_pred_tt = logreg.predict(x_tt)
    logreg_acc_tt_aux = np.mean(logreg_pred_tt == y_tt)
    assert np.isclose(logreg_acc_tt, logreg_acc_tt_aux)
    print("Baseline - Train accuracy: {:.4f}".format(baseline_acc_tr))
    print("Baseline - Test accuracy: {:.4f}".format(baseline_acc_tt))
    print("LogReg - Train accuracy: {:.4f}".format(logreg_acc_tr))
    print("LogReg - Test accuracy: {:.4f}".format(logreg_acc_tt))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
