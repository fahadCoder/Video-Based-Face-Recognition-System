import numpy as np
from time import perf_counter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


from cvproj_exc.osr_learning import (
    UNKNOWN_LABEL,
    spl_training,
    mpl_training,
    load_challenge_train_data,
)

UNKNOWN = UNKNOWN_LABEL


def make_uuc_split(x, y, uuc_frac=0.2, val_kuc_frac=0.5, seed=0):
  
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    kc_mask = (y >= 0)
    kuc_mask = (y == UNKNOWN_LABEL)

    # choose some KC identities to act as "UUC"
    kc_labels = np.unique(y[kc_mask])
    uuc_labels = rng.choice(
        kc_labels, size=max(1, int(len(kc_labels) * uuc_frac)), replace=False
    )

    uuc_mask = kc_mask & np.isin(y, uuc_labels)
    remain_kc_mask = kc_mask & (~uuc_mask)

    # --- split remaining KCs: take 1 sample per class into val-known ---
    x_kc = x[remain_kc_mask]
    y_kc = y[remain_kc_mask]

    train_idx = []
    val_idx = []
    for lab in np.unique(y_kc):
        idx = np.flatnonzero(y_kc == lab)
        rng.shuffle(idx)
        if idx.size >= 2:
            val_idx.append(idx[0])          # 1 sample to validation as known
            train_idx.extend(idx[1:])       # rest stay in training
        else:
            train_idx.extend(idx)           # singleton classes stay only in training

    train_idx = np.array(train_idx, dtype=int)
    val_idx = np.array(val_idx, dtype=int)

    x_kc_tr, y_kc_tr = x_kc[train_idx], y_kc[train_idx]
    x_kc_va, y_kc_va = x_kc[val_idx], y_kc[val_idx]

    # --- split KUC into train/val (optional, but helps AUC/FAR sanity) ---
    x_kuc = x[kuc_mask]
    y_kuc = y[kuc_mask]
    if x_kuc.shape[0] > 0:
        perm = rng.permutation(x_kuc.shape[0])
        cut = int((1.0 - val_kuc_frac) * x_kuc.shape[0])
        tr_ids, va_ids = perm[:cut], perm[cut:]
        x_kuc_tr, y_kuc_tr = x_kuc[tr_ids], y_kuc[tr_ids]
        x_kuc_va, y_kuc_va = x_kuc[va_ids], y_kuc[va_ids]
    else:
        x_kuc_tr = x_kuc_va = np.empty((0, x.shape[1]), dtype=np.float32)
        y_kuc_tr = y_kuc_va = np.empty((0,), dtype=int)

    # --- withheld UUC identities go to val as UNKNOWN ---
    x_uuc_va = x[uuc_mask]
    y_uuc_va = np.full((x_uuc_va.shape[0],), UNKNOWN_LABEL, dtype=int)

    # build train/val
    x_tr = np.concatenate([x_kc_tr, x_kuc_tr], axis=0)
    y_tr = np.concatenate([y_kc_tr, y_kuc_tr], axis=0)

    x_va = np.concatenate([x_kc_va, x_kuc_va, x_uuc_va], axis=0)
    y_va = np.concatenate([y_kc_va, y_kuc_va, y_uuc_va], axis=0)

    # shuffle val
    perm = rng.permutation(x_va.shape[0])
    x_va, y_va = x_va[perm], y_va[perm]

    print("VAL known:", int(np.sum(y_va >= 0)), "VAL unknown:", int(np.sum(y_va == UNKNOWN_LABEL)))
    return x_tr, y_tr, x_va, y_va


def threshold_at_far(scores_unknown, far):
    # FAR = P(score >= tau | unknown) => tau = quantile(1 - FAR) of unknown scores
    return float(np.quantile(scores_unknown, 1.0 - far))


def dir_at_far(y_true, y_pred, y_score, far):
    known_mask = (y_true >= 0)
    unk_mask = ~known_mask

    if unk_mask.sum() == 0 or known_mask.sum() == 0:
        return np.nan, np.nan

    tau = threshold_at_far(y_score[unk_mask], far)
    accepted = known_mask & (y_score >= tau)
    correct = (y_pred == y_true)

    dir_val = (accepted & correct).sum() / known_mask.sum()
    return float(dir_val), tau


def balanced_rank1(y_true, y_pred):
    kc = y_true[y_true >= 0]
    if kc.size == 0:
        return np.nan
    labels = np.unique(kc)
    accs = []
    for lab in labels:
        m = (y_true == lab)
        accs.append((y_pred[m] == lab).mean())
    return float(np.mean(accs))


def evaluate(predict_fn, x_val, y_val):
    y_pred, y_score = predict_fn(x_val)

    # AUCROC: known=1 vs unknown=0
    is_known = (y_val >= 0).astype(int)
    auc = roc_auc_score(is_known, y_score)

    dir1, tau1 = dir_at_far(y_val, y_pred, y_score, far=0.01)
    dir10, tau10 = dir_at_far(y_val, y_pred, y_score, far=0.10)

    bal_r1 = balanced_rank1(y_val, y_pred)

    return {
        "AUCROC": auc,
        "DIR@FAR=1%": dir1,
        "DIR@FAR=10%": dir10,
        "Balanced Rank-1": bal_r1,
        "tau@1%": tau1,
        "tau@10%": tau10,
    }


def time_fit(train_fn, x_tr, y_tr, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        _ = train_fn(x_tr, y_tr)
        times.append(perf_counter() - t0)
    return float(np.median(times))


def time_predict(predict_fn, x, repeats=5):
    # warmup
    predict_fn(x[: min(64, len(x))])

    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        predict_fn(x)
        times.append(perf_counter() - t0)

    t = float(np.median(times))
    return t / len(x)


def main():
    x, y = load_challenge_train_data()
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    # simulate unknown-unknowns by withholding some known identities
    x_tr, y_tr, x_va, y_va = make_uuc_split(x, y, uuc_frac=0.2, seed=0)

    # SPL
    t_fit_spl = time_fit(spl_training, x_tr, y_tr)
    spl_fn = spl_training(x_tr, y_tr)
    t_pred_spl = time_predict(spl_fn, x_va)
    metrics_spl = evaluate(spl_fn, x_va, y_va)

    # MPL
    t_fit_mpl = time_fit(mpl_training, x_tr, y_tr)
    mpl_fn = mpl_training(x_tr, y_tr)
    t_pred_mpl = time_predict(mpl_fn, x_va)
    metrics_mpl = evaluate(mpl_fn, x_va, y_va)

    print("\n=== SPL ===")
    for k, v in metrics_spl.items():
        print(f"{k}: {v}")
    print(f"Fit time (s): {t_fit_spl:.4f}")
    print(f"Predict time per sample (s): {t_pred_spl:.8f}")

    print("\n=== MPL ===")
    for k, v in metrics_mpl.items():
        print(f"{k}: {v}")
    print(f"Fit time (s): {t_fit_mpl:.4f}")
    print(f"Predict time per sample (s): {t_pred_mpl:.8f}")


if __name__ == "__main__":
    main()
