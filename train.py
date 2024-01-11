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
    parser.add_argument(
        "--do_normalize",
        action="store_true",
        help="Normalize train and test data",
    )
    parser.add_argument(
        "--do_print",
        action="store_true",
        help="Print information during execution",
    )
    return parser


def read_data(
    train_path, test_path, cols_exclude=["lat", "lon", "time", "LABELS"], do_print=True
):
    df_tr = pd.read_csv(train_path, index_col=0)
    df_tt = pd.read_csv(test_path, index_col=0)
    features = [f for f in df_tr.keys() if f not in cols_exclude]
    x_tr = df_tr[features].values
    x_tt = df_tt[features].values
    y_tr = df_tr.LABELS.values
    y_tt = df_tt.LABELS.values
    if do_print:
        print(f"Number of features: {len(features)}")
        print(f"Number of classes: {np.unique(y_tr).shape[0]}")
        print(f"Train size: {x_tr.shape[0]}")
        print(f"Test size: {x_tt.shape[0]}")
    return x_tr, x_tt, y_tr, y_tt


def normalize(x_tr, x_tt):
    mean_tr = np.mean(x_tr, axis=0)
    std_tr = np.std(x_tr, axis=0)
    x_tr = (x_tr - mean_tr) / std_tr
    x_tt = (x_tt - mean_tr) / std_tr
    return x_tr, x_tt


def logistic_regression_train(x_tr, y_tr, n_iters=500, seed=0):
    return LogisticRegression(max_iter=n_iters, random_state=seed).fit(x_tr, y_tr)


def most_frequent_train(x_tr, y_tr):
    return DummyClassifier(strategy="most_frequent").fit(x_tr, y_tr)


def evaluate_accuracy(model, x, y):
    return model.score(x, y)


def main(args):
    # Read data
    if not args.train:
        print("Train data file missing")
        return
    if not args.test:
        print("Test data file missing")
        return
    x_tr, x_tt, y_tr, y_tt = read_data(args.train, args.test, do_print=args.do_print)

    # Normalize
    if args.do_normalize:
        x_tr, x_tt = normalize(x_tr, x_tt)

    # Train models
    baseline = most_frequent_train(x_tr, y_tr)
    logreg = logistic_regression_train(x_tr, y_tr, n_iters=500, seed=0)

    # Evaluation
    baseline_acc_tr = evaluate_accuracy(baseline, x_tr, y_tr)
    baseline_acc_tt = evaluate_accuracy(baseline, x_tt, y_tt)
    logreg_acc_tt = evaluate_accuracy(logreg, x_tt, y_tt)
    logreg_acc_tr = evaluate_accuracy(logreg, x_tr, y_tr)
    print("Baseline - Train accuracy: {:.4f}".format(baseline_acc_tr))
    print("Baseline - Test accuracy: {:.4f}".format(baseline_acc_tt))
    print("LogReg - Train accuracy: {:.4f}".format(logreg_acc_tr))
    print("LogReg - Test accuracy: {:.4f}".format(logreg_acc_tt))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
