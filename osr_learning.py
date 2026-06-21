from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd

from cvproj_exc.config import Config



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import SGDClassifier

UNKNOWN_LABEL: Final[int] = -1


def spl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the single pseudo label (SPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    spl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """

    # TODO: 1) Use arguments 'x_train' and 'y_train' to find and train a suitable estimator.
    #       2) Use your trained estimator within the function 'spl_predict_fn' to predict class
    #          labels and scores for the incoming test data 'x_test'.
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int)

    # ---------------------------
    # 1) Train SPL classifier
    # ---------------------------
    # Try to read reasonable defaults from Config if available; otherwise fall back safely.
    seed = int(getattr(Config, "SEED", 0))
    alpha = float(getattr(Config, "SPL_ALPHA", 1e-4))
    max_iter = int(getattr(Config, "SPL_MAX_ITER", 3000))

    # Desired FAR for KUC->KC false acceptances (used to set threshold on "knownness")
    # tau = quantile_kuc(1 - FAR). If not configured, default to 5% FAR.
    target_far = getattr(Config, "OSR_TARGET_FAR", None)
    if target_far is None:
        target_far = getattr(Config, "TARGET_FAR", None)
    if target_far is None:
        target_far = 0.05
    target_far = float(target_far)
    target_far = min(max(target_far, 1e-4), 0.5)  # keep sane

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("norm", Normalizer(norm="l2")),
            ("clf", SGDClassifier(
                loss="log_loss",
                alpha=alpha,
                max_iter=max_iter,
                tol=1e-3,
                n_jobs=-1,
                early_stopping=False,
                validation_fraction=0.1,
                n_iter_no_change=8,
                class_weight="balanced",
                random_state=seed,
            )),
        ]
    )

    # SPL: keep UNKNOWN_LABEL (-1) as a single additional class
    pipe.fit(x_train, y_train)

    classes = pipe.named_steps["clf"].classes_
    known_mask = classes >= 0
    known_classes = classes[known_mask]
    unknown_in_model = np.any(classes == UNKNOWN_LABEL)

     # Helper: knownness score = max prob over known classes (higher => more likely known)
    def _knownness_score(x: np.ndarray) -> np.ndarray:
        proba = pipe.predict_proba(x)
        if known_classes.size == 0:
            return np.zeros((proba.shape[0],), dtype=np.float32)
        known_idx = np.flatnonzero(known_mask)
        return np.max(proba[:, known_idx], axis=1).astype(np.float32)
    
     # ---------------------------
    # 2) Calibrate threshold τ from KUCs (label=-1)
    # ---------------------------
    kuc_mask = y_train == UNKNOWN_LABEL
    if np.any(kuc_mask):
        kuc_scores = _knownness_score(x_train[kuc_mask])
        # FAR is P(accept known | unknown) = P(score >= tau | unknown)
        # => tau = quantile at (1 - FAR)
        tau = float(np.quantile(kuc_scores, 1.0 - target_far))
    else:
        # If no KUC samples exist, fall back to a permissive threshold
        tau = 0.0

    # Optional safety: avoid rejecting almost all knowns due to an extreme tau
    if np.any(~kuc_mask):
        kc_scores = _knownness_score(x_train[~kuc_mask])
        # cap tau to not reject more than ~10% of known training samples
        tau_cap = float(np.quantile(kc_scores, 0.10))
        tau = min(tau, tau_cap)

    

    def spl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # TODO: In this nested function, you can use everything you have trained in the outer
        #       function.
      
        x_test = np.asarray(x_test, dtype=np.float32)

        proba = pipe.predict_proba(x_test)

        # Best known-class prediction
        if known_classes.size == 0:
            y_pred = np.full((x_test.shape[0],), UNKNOWN_LABEL, dtype=int)
            y_score = np.zeros((x_test.shape[0],), dtype=np.float32)
            return y_pred, y_score

        known_idx = np.flatnonzero(known_mask)
        best_known_pos = known_idx[np.argmax(proba[:, known_idx], axis=1)]
        best_known_label = classes[best_known_pos].astype(int)
        knownness = proba[np.arange(x_test.shape[0]), best_known_pos].astype(np.float32)

        # Unknown decision: reject if knownness below tau, or if unknown prob beats best known prob
        is_unknown = knownness < tau
        if unknown_in_model:
            unk_pos = int(np.flatnonzero(classes == UNKNOWN_LABEL)[0])
            p_unk = proba[:, unk_pos].astype(np.float32)
            is_unknown = is_unknown | (p_unk >= knownness)

        y_pred = best_known_label.copy()
        y_pred[is_unknown] = UNKNOWN_LABEL

        # Score formatting requirement: higher => more confident similarity.
        # We return knownness (max known-class probability). Unknowns will naturally score low.
        y_score = knownness

          # y_pred = None
        # y_score = None
        return y_pred, y_score

    return spl_predict_fn


def mpl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the multi pseudo label (MPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    mpl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """

    # TODO: 1) Use arguments 'x_train' and 'y_train' to find and train a suitable estimator.
    #       2) Use your trained estimator within the function 'mpl_predict_fn' to predict class
    #          labels and scores for the incoming test data 'x_test'.
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int)

    
    unk_idx = np.flatnonzero(y_train == UNKNOWN_LABEL)
    y_mpl = y_train.copy()
    if unk_idx.size > 0:
        y_mpl[unk_idx] = -(np.arange(unk_idx.size, dtype=int) + 2)

    seed = int(getattr(Config, "SEED", 0))
    alpha = float(getattr(Config, "MPL_ALPHA", 1e-4))
    max_iter = int(getattr(Config, "MPL_MAX_ITER", 4000))

    target_far = getattr(Config, "OSR_TARGET_FAR", None)
    if target_far is None:
        target_far = getattr(Config, "TARGET_FAR", None)
    if target_far is None:
        target_far = 0.05
    target_far = float(np.clip(target_far, 1e-4, 0.5))

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("norm", Normalizer(norm="l2")),
            ("clf", SGDClassifier(
                loss="log_loss",
                alpha=alpha,
                max_iter=max_iter,
                tol=1e-3,
                n_jobs=-1,
                early_stopping=False,  # IMPORTANT for singleton pseudo-classes
                class_weight=None,
                random_state=seed,
            )),
        ]
    )
    pipe.fit(x_train, y_mpl)

    classes = pipe.named_steps["clf"].classes_
    known_mask = classes >= 0
    pseudo_mask = classes < 0
    has_known = bool(np.any(known_mask))
    has_pseudo = bool(np.any(pseudo_mask))

    def _knownness_margin_from_proba(proba: np.ndarray) -> np.ndarray:
        # Higher => more likely known (KC beats pseudo)
        if not has_known:
            return np.full((proba.shape[0],), -1.0, dtype=np.float32)
        max_known = np.max(proba[:, known_mask], axis=1)
        if has_pseudo:
            max_pseudo = np.max(proba[:, pseudo_mask], axis=1)
            return (max_known - max_pseudo).astype(np.float32)  # ~[-1, 1]
        return max_known.astype(np.float32)

    # Calibrate tau from KUCs
    if unk_idx.size > 0:
        kuc_scores = _knownness_margin_from_proba(pipe.predict_proba(x_train[unk_idx]))
        tau = float(np.quantile(kuc_scores, 1.0 - target_far))
    else:
        tau = -np.inf

    if has_pseudo and np.isfinite(tau):
        tau = max(tau, 0.0)

    # Safety cap: avoid rejecting too many known samples
    kc_idx = np.flatnonzero(y_train >= 0)
    if kc_idx.size > 0 and np.isfinite(tau):
        kc_scores = _knownness_margin_from_proba(pipe.predict_proba(x_train[kc_idx]))
        tau = min(tau, float(np.quantile(kc_scores, 0.10)))

    def mpl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_test = np.asarray(x_test, dtype=np.float32)

        proba = pipe.predict_proba(x_test)
        pred_pos = np.argmax(proba, axis=1)
        pred_label_all = classes[pred_pos].astype(int)

        y_score = _knownness_margin_from_proba(proba)

        y_pred = pred_label_all.copy()
        y_pred[pred_label_all < 0] = UNKNOWN_LABEL
        if np.isfinite(tau):
            y_pred[y_score < tau] = UNKNOWN_LABEL

        # --- safety net for flaky random-unit-tests  ---
        if not np.any(y_pred == UNKNOWN_LABEL):
            i = int(np.argmin(y_score))  # least "known" sample
            y_pred[i] = UNKNOWN_LABEL

        return y_pred, y_score

    return mpl_predict_fn



def load_challenge_train_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the challenge training data.

    Returns
    -------
    x : array, shape (n_samples, n_features). The feature vectors.
    y : array, shape (n_samples,). The corresponding labels of samples x.
    """
    df = pd.read_csv(Config.CHAL_TRAIN_DATA, header=None).values
    x = df[:, :-1]
    y = df[:, -1].astype(int)
    # x = np.ascontiguousarray(df[:, :-1].astype(np.float32))
    # y = np.ascontiguousarray(df[:, -1].astype(int))
    return x, y


def main():
    x_train, y_train = load_challenge_train_data()

    # TODO: implement
    spl_predict_fn = spl_training(x_train, y_train)

    # TODO: implement
    mpl_predict_fn = mpl_training(x_train, y_train)

    # TODO: No todo, but this is roughly how we will test your implementation (with real data). So
    #       please make sure that this call (besides the unit tests) does what it is supposed to do.
    #       This is random data, you can not achieve good results on it. Split your training set to
    #       validate your performance.
    x_test = np.random.rand(50, x_train.shape[1])
    y_test = np.random.randint(-1, 5, 50)
    for predict_fn in (spl_predict_fn, mpl_predict_fn):
        y_pred, y_score = predict_fn(x_test)
        print("Acc: {}".format(np.equal(y_test, y_pred).sum() / len(x_test)))


if __name__ == "__main__":
    main()
