"""
A script to train a machine learning model on ClimateNet
"""
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def add_args(parser):
    """
    Add arguments to parser
    """
    parser.add_argument(
        "--train",
        default="data/train.csv",
        type=str,
        help="Train CSV",
    )
    parser.add_argument(
        "--test",
        default="data/test.csv",
        type=str,
        help="Test CSV",
    )
    parser.add_argument(
        "--pct_val",
        default=0.0,
        type=float,
        help="Percentage of the train set for the validation set.",
    )
    parser.add_argument(
        "--pct_test",
        default=0.0,
        type=float,
        help="Percentage of the train set for the test set. If larger than 0, the "
        "original test set is reserved (ignored).",
    )
    parser.add_argument(
        "--do_normalize",
        action="store_true",
        help="Normalize train and test data",
    )
    parser.add_argument(
        "--n_iters",
        default=500,
        type=int,
        help="Number of gradient descent iterations",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--do_hp_validation",
        action="store_true",
        help="Do hyper-parameter optimisation with the validation set",
    )
    parser.add_argument(
        "--do_balanced_class_weights",
        action="store_true",
        help="Use balanced class weights in the loss function",
    )
    parser.add_argument(
        "--do_print",
        action="store_true",
        help="Print information during execution",
    )
    return parser


def read_data(train_path, test_path, cols_exclude=["lat", "lon", "time", "LABELS"]):
    df_tr = pd.read_csv(train_path, index_col=0)
    df_tt = pd.read_csv(test_path, index_col=0)
    features = [f for f in df_tr.keys() if f not in cols_exclude]
    x_tr = df_tr[features].values
    x_tt = df_tt[features].values
    y_tr = df_tr.LABELS.values
    y_tt = df_tt.LABELS.values
    return x_tr, x_tt, y_tr, y_tt


def split(x_tr, y_tr, pct_val, pct_tt, seed=0):
    n_val = int(pct_val * len(y_tr))
    n_tt = int(pct_tt * len(y_tr))
    # Validation split
    if n_val > 0:
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_tr, y_tr, test_size=n_val, shuffle=True, stratify=y_tr
        )
    else:
        x_val = []
        y_val = []
    # Test split
    if n_tt > 0:
        x_tr, x_tt, y_tr, y_tt = train_test_split(
            x_tr, y_tr, test_size=n_tt, shuffle=True, stratify=y_tr
        )
    else:
        x_tt = []
        y_tt = []
    return x_tr, x_val, x_tt, y_tr, y_val, y_tt


def normalize(x_tr, x_val, x_tt):
    mean_tr = np.mean(x_tr, axis=0)
    std_tr = np.std(x_tr, axis=0)
    x_tr = (x_tr - mean_tr) / std_tr
    if len(x_val) > 0:
        x_val = (x_val - mean_tr) / std_tr
    if len(x_tt) > 0:
        x_tt = (x_tt - mean_tr) / std_tr
    return x_tr, x_val, x_tt


def logistic_regression_train(x_tr, y_tr, n_iters=500, do_balanced=False, seed=0):
    if do_balanced:
        class_weight = "balanced"
    else:
        class_weight = None
    return LogisticRegression(
        max_iter=n_iters, class_weight=class_weight, random_state=seed
    ).fit(x_tr, y_tr)


def logistic_regression_validate_reg_train(
    x_tr, y_tr, x_val, y_val, c_values, n_iters=500, do_balanced=False, seed=0
):
    if do_balanced:
        class_weight = "balanced"
    else:
        class_weight = None
    models = []
    for c in c_values:
        models.append(
            LogisticRegression(
                C=c, max_iter=n_iters, class_weight=class_weight, random_state=seed
            ).fit(x_tr, y_tr)
        )
    acc = [evaluate_accuracy(model, x_val, y_val) for model in models]
    return models[np.argmax(acc)]


def most_frequent_train(x_tr, y_tr):
    return DummyClassifier(strategy="most_frequent").fit(x_tr, y_tr)


def evaluate_accuracy(model, x, y):
    return model.score(x, y)


def main(args):
    # Read data
    if not args.train:
        print("Train data file missing. Training will not proceed.")
        return
    if not args.test:
        print("Test data file missing. Training will not proceed.")
        return
    x_tr, x_tt_orig, y_tr, y_tt_orig = read_data(args.train, args.test)
    if args.do_print:
        print(f"Number of features: {x_tr.shape[1]}")
        print(f"Number of classes: {np.unique(y_tr).shape[0]}")
    # Split
    x_tr, x_val, x_tt, y_tr, y_val, y_tt = split(
        x_tr, y_tr, args.pct_val, args.pct_test, seed=args.seed
    )
    if len(y_tt) == 0:
        x_tt = x_tt_orig
        y_tt = y_tt_orig
    if args.do_print:
        print(f"Train size: {len(x_tr)}")
        print(f"Validation size: {len(x_val)}")
        print(f"Test size: {len(x_tt)}")

    # Normalize
    if args.do_normalize:
        x_tr, x_val, x_tt = normalize(x_tr, x_val, x_tt)

    # Train models
    baseline = most_frequent_train(x_tr, y_tr)
    if args.do_hp_validation and len(x_val) > 0:
        c_values = 10.0 ** np.arange(-2, 3)
        logreg = logistic_regression_validate_reg_train(
            x_tr,
            y_tr,
            x_val,
            y_val,
            c_values,
            n_iters=args.n_iters,
            do_balanced=args.do_balanced_class_weights,
            seed=args.seed,
        )
    else:
        logreg = logistic_regression_train(
            x_tr,
            y_tr,
            n_iters=args.n_iters,
            do_balanced=args.do_balanced_class_weights,
            seed=args.seed,
        )

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
