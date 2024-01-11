import sys

import pytest

sys.path.append('../')
import numpy as np

from train import normalize, read_data

train_path = "./data/train.csv"
test_path = "./data/test.csv"

def test__read_data__returns_reasonable_structures():
    x_tr, x_tt, y_tr, y_tt = read_data(train_path, test_path, do_print=False)
    # Check that returned types are np.ndarray
    assert isinstance(x_tr, np.ndarray)
    assert isinstance(x_tt, np.ndarray)
    assert isinstance(y_tr, np.ndarray)
    assert isinstance(y_tt, np.ndarray)
    # Check that dimensions are reasonable
    assert x_tr.shape[1] == x_tt.shape[1]
    assert y_tr.ndim == 1
    assert y_tt.ndim == 1

def test__normalize__yields_expected_statistics():
    x_tr, x_tt, y_tr, y_tt = read_data(train_path, test_path, do_print=False)
    x_tr, x_tt = normalize(x_tr, x_tt)
    assert all(np.isclose(x_tr.mean(axis=0), np.zeros(x_tr.shape[1])))
