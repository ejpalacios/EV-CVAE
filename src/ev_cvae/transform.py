import numpy as np


def scale(Xs, mu, sigma):
    return (Xs - mu) / sigma


def inv_scale(Xs, mu, sigma):
    return Xs * sigma + mu


def log1p_cols(X, x_cols, pos_cols):
    X = X.copy()
    pos_idx = [x_cols.index(c) for c in pos_cols]
    X[:, pos_idx] = np.log1p(np.clip(X[:, pos_idx], 0, None))
    return X


def inv_log1p_cols(X, x_cols, pos_cols):
    X = X.copy()
    pos_idx = [x_cols.index(c) for c in pos_cols]
    X[:, pos_idx] = np.expm1(X[:, pos_idx])
    return X
