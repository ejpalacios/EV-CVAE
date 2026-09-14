import numpy as np
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# -------------------------
# 1) Wasserstein metrics
# -------------------------
def w1_per_feature(real, synth):
    return np.array(
        [wasserstein_distance(real[:, d], synth[:, d]) for d in range(real.shape[1])]
    )


def sliced_wasserstein(real, synth, n_proj=200, seed=0):
    rng = np.random.default_rng(seed)
    D = real.shape[1]
    dirs = rng.normal(size=(n_proj, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    vals = []
    for w in dirs:
        vals.append(wasserstein_distance(real @ w, synth @ w))
    return float(np.mean(vals))


# -------------------------
# 2) Spearman corr structure
# -------------------------
def spearman_matrix(x):
    D = x.shape[1]
    R = np.zeros((D, D), dtype=np.float32)
    for i in range(D):
        for j in range(D):
            R[i, j] = spearmanr(x[:, i], x[:, j]).correlation
    return np.nan_to_num(R, nan=0.0)


def corr_mae(real, synth):
    Rr = spearman_matrix(real)
    Rs = spearman_matrix(synth)
    return float(np.mean(np.abs(Rr - Rs)))


# -------------------------
# 3) ACF MAE (build sequences by sorting within each condition)
#    Uses WeekTime_sin/cos to define an ordering in the week.
# -------------------------
def acf_1d(x, max_lag):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < max_lag + 2:
        return None
    x = x - x.mean()
    denom = np.dot(x, x) + 1e-12
    acf = np.empty(max_lag + 1, dtype=np.float64)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.dot(x[:-lag], x[lag:]) / denom
    return acf


def build_sorted_sequence(X_scaled_block, x_cols):
    # sort by week-angle inferred from sin/cos
    si = x_cols.index("WeekTime_sin")
    co = x_cols.index("WeekTime_cos")
    ang = np.arctan2(X_scaled_block[:, si], X_scaled_block[:, co])  # [-pi, pi]
    order = np.argsort(ang)
    return X_scaled_block[order]


def mean_acf_over_blocks(X, day_arr, managed_arr, x_cols, feature, max_lag=24):
    fi = x_cols.index(feature)
    acfs = []
    for d in range(7):
        for m in [0, 1]:
            mask = (day_arr == d) & (managed_arr == m)
            block = X[mask]
            if block.shape[0] < max_lag + 2:
                continue
            block = build_sorted_sequence(block, x_cols)
            a = acf_1d(block[:, fi], max_lag=max_lag)
            if a is not None:
                acfs.append(a)
    if not acfs:
        return None
    return np.mean(np.stack(acfs, axis=0), axis=0)


def acf_mae(
    X_real,
    X_synth,
    C_real,
    C_synth,
    x_cols,
    feature="ConsumedkWh",
    max_lag=24,
):
    day_real = np.argmax(C_real[:, :7], axis=1).astype(int)
    man_real = C_real[:, 7].round().astype(int)
    ar = mean_acf_over_blocks(
        X_real, day_real, man_real, x_cols, feature, max_lag=max_lag
    )

    day_s = np.argmax(C_synth[:, :7], axis=1).astype(int)
    man_s = C_synth[:, 7].round().astype(int)
    as_ = mean_acf_over_blocks(X_synth, day_s, man_s, x_cols, feature, max_lag=max_lag)

    if ar is None or as_ is None:
        return None
    return float(np.mean(np.abs(ar - as_)))


# -------------------------
# 4) TSTR utility (regression)
#    Default: predict ConsumedkWh from the other features + condition
# -------------------------
def tstr_regression(
    X_real_train,
    C_real_train,
    X_real_test,
    C_real_test,
    X_synth,
    C_synth,
    x_cols,
    feature="ConsumedkWh",
    seed=0,
):
    tgt_i = x_cols.index(feature)

    # Real train data
    Xr_tr = np.concatenate(
        [np.delete(X_real_train, tgt_i, axis=1), C_real_train], axis=1
    )
    yr_tr = X_real_train[:, tgt_i]

    # Synthetic train data
    Xs = np.concatenate([np.delete(X_synth, tgt_i, axis=1), C_synth], axis=1)
    ys = X_synth[:, tgt_i]

    # Real test data
    Xr_te = np.concatenate([np.delete(X_real_test, tgt_i, axis=1), C_real_test], axis=1)
    yr_te = X_real_test[:, tgt_i]

    m_syn = RandomForestRegressor(random_state=seed, n_estimators=200, n_jobs=-1)
    m_syn.fit(Xs, ys)
    pred_tstr = m_syn.predict(Xr_te)
    mae_tstr = mean_absolute_error(yr_te, pred_tstr)

    m_real = RandomForestRegressor(random_state=seed, n_estimators=200, n_jobs=-1)
    m_real.fit(Xr_tr, yr_tr)
    pred_trtr = m_real.predict(Xr_te)
    mae_trtr = mean_absolute_error(yr_te, pred_trtr)

    return (
        {
            "MAE_TSTR_scaled": float(mae_tstr),
            "MAE_TRTR_scaled": float(mae_trtr),
            "ratio": float(mae_tstr / (mae_trtr + 1e-12)),
        },
        m_real,
        m_syn,
    )


def tstr_prediction(
    m_real, m_syn, X_real_test, C_real_test, x_cols, feature="ConsumedKWh"
):
    tgt_i = x_cols.index(feature)

    # Real test data
    Xr_te = np.concatenate([np.delete(X_real_test, tgt_i, axis=1), C_real_test], axis=1)
    yr_te = X_real_test[:, tgt_i]
    pred_tstr = m_syn.predict(Xr_te)
    mae_tstr = mean_absolute_error(yr_te, pred_tstr)

    pred_trtr = m_real.predict(Xr_te)
    mae_trtr = mean_absolute_error(yr_te, pred_trtr)
    return {
        "MAE_TSTR_scaled": float(mae_tstr),
        "MAE_TRTR_scaled": float(mae_trtr),
        "ratio": float(mae_tstr / (mae_trtr + 1e-12)),
    }
