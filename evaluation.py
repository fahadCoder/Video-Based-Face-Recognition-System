import pickle

import numpy as np

from cvproj_exc.classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(
        self,
        classifier=NearestNeighborClassifier(),
        false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True),
    ):
        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []
 
        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):
        with open(train_data_file, "rb") as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding="bytes")
        with open(test_data_file, "rb") as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding="bytes")

    # Run the evaluation and find performance measure (identification rates) at different
    # similarity thresholds.
    def run(self):
        # similarity_thresholds = None
        # identification_rates = None

        # # Report all performance measures.
        # evaluation_results = {
        #     "similarity_thresholds": similarity_thresholds,
        #     "identification_rates": identification_rates,
        # }
        # --- Fit on training data ---
        train_emb = np.asarray(self.train_embeddings)
        train_lbl = np.asarray(self.train_labels)
        self.classifier.fit(train_emb, train_lbl)

        # --- Predict on test data (closed-set) ---
        test_emb = np.asarray(self.test_embeddings)
        pred_labels, similarities = self.classifier.predict_labels_and_similarities(test_emb)

        pred_labels = np.asarray(pred_labels).copy()
        similarities = np.asarray(similarities)

        # --- Sweep FARs and compute DIR / thresholds ---
        similarity_thresholds = np.zeros_like(self.false_alarm_rate_range, dtype=float)
        identification_rates = np.zeros_like(self.false_alarm_rate_range, dtype=float)

        for i, far in enumerate(self.false_alarm_rate_range):
            tau = self.select_similarity_threshold(similarities, far)

            # Open-set decision: reject as unknown if similarity < tau
            open_set_labels = pred_labels.copy()
            open_set_labels[similarities < tau] = UNKNOWN_LABEL

            similarity_thresholds[i] = tau
            identification_rates[i] = self.calc_identification_rate(open_set_labels)

        evaluation_results = {
            "false_alarm_rates": np.asarray(self.false_alarm_rate_range, dtype=float),
            "similarity_thresholds": similarity_thresholds,
            "identification_rates": identification_rates,
        }

        return evaluation_results

    def select_similarity_threshold(self, similarity, false_alarm_rate):
        """
        Choose threshold tau so that the false alarm rate (FAR) on *unknown* probes is ~false_alarm_rate.

        FAR(alpha) means: fraction of unknown probes incorrectly accepted as known:
            FAR = P(similarity >= tau | unknown)

        So tau is the (1 - alpha)-quantile of the unknown similarities:
            tau = percentile(unknown_sim, 100*(1-alpha))
        """
        similarity = np.asarray(similarity)
        test_lbl = np.asarray(self.test_labels)

        unknown_mask = (test_lbl == UNKNOWN_LABEL)
        unknown_sim = similarity[unknown_mask]

        if unknown_sim.size == 0:
            raise ValueError("No unknown samples in test set; cannot estimate FAR threshold.")

        # Clamp FAR to [0, 1] to avoid percentile errors if caller passes odd values
        alpha = float(np.clip(false_alarm_rate, 0.0, 1.0))
        p = 100.0 * (1.0 - alpha)

        return float(np.percentile(unknown_sim, p))


    def calc_identification_rate(self, prediction_labels):
          
        # Rank-1 identification rate (DIR at rank-1): fraction of *known* probes correctly identified.
        # Unknown probes are not part of the denominator.
        
        pred = np.asarray(prediction_labels)
        gt = np.asarray(self.test_labels)

        known_mask = (gt != UNKNOWN_LABEL)
        if np.sum(known_mask) == 0:
            return 0.0

        return float(np.mean(pred[known_mask] == gt[known_mask]))






# import pickle

# import numpy as np

# from cvproj_exc.classifier import NearestNeighborClassifier

# # Class label for unknown subjects in test and training data.
# UNKNOWN_LABEL = -1


# # Evaluation of open-set face identification.
# class OpenSetEvaluation:

#     def __init__(
#         self,
#         classifier=NearestNeighborClassifier(),
#         false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True),
#     ):
#         # The false alarm rates.
#         self.false_alarm_rate_range = false_alarm_rate_range

#         # Datasets (embeddings + labels) used for training and testing.
#         self.train_embeddings = []
#         self.train_labels = []
#         self.test_embeddings = []
#         self.test_labels = []

#         # The evaluated classifier (see classifier.py)
#         self.classifier = classifier

#     # Prepare the evaluation by reading training and test data from file.
#     def prepare_input_data(self, train_data_file, test_data_file):
#         with open(train_data_file, "rb") as f:
#             (self.train_embeddings, self.train_labels) = pickle.load(f, encoding="bytes")
#         with open(test_data_file, "rb") as f:
#             (self.test_embeddings, self.test_labels) = pickle.load(f, encoding="bytes")

#     # Run the evaluation and find performance measure (identification rates) at different
#     # similarity thresholds.
#     def run(self):
#         similarity_thresholds = None
#         identification_rates = None

#         # Report all performance measures.
#         evaluation_results = {
#             "similarity_thresholds": similarity_thresholds,
#             "identification_rates": identification_rates,
#         }

#         return evaluation_results

#     def select_similarity_threshold(self, similarity, false_alarm_rate):
#         return None

#     def calc_identification_rate(self, prediction_labels):
#         return None
