import os
import pickle
import cv2
import numpy as np
from cvproj_exc.config import Config

# FaceNet to extract face embeddings.
class FaceNet:
    def __init__(self):
        self.facenet = cv2.dnn.readNetFromONNX(str(Config.RESNET50))
        self.facenet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.facenet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.input_size = (224, 224)
        self.rgb_mean = np.array([131.0912, 103.8827, 91.4953], dtype=np.float32)

    def predict(self, face: np.ndarray) -> np.ndarray:
        if face is None or face.size == 0:
            raise ValueError("FaceNet.predict received an empty face image.")

        if (face.shape[1], face.shape[0]) != self.input_size:
            face = cv2.resize(face, self.input_size, interpolation=cv2.INTER_LINEAR)

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        face_rgb -= self.rgb_mean

        blob = np.transpose(face_rgb, (2, 0, 1))[None, :, :, :]
        self.facenet.setInput(blob)
        embedding = np.squeeze(self.facenet.forward()).astype(np.float32)

        norm = float(np.linalg.norm(embedding) + 1e-12)
        return embedding / norm

    @classmethod
    @property
    def embedding_dimensionality(cls):
        return 128


class FaceRecognizer:
    # # 11,0.8.0.5 -> not good result
    # 11,0.95,0.5 -> good result
    # 7,0.95,0.5 -> not good result
    # 9,0.95,0.5 -> not good but better then others
    # num_neighbours=13, max_distance=1.05, min_prob=0.5   best

    def __init__(self, num_neighbours=13, max_distance=1.05, min_prob=0.4): 
        # Prepare FaceNet and set all parameters for kNN.
        self.facenet = FaceNet()
        self.num_neighbours = int(num_neighbours)

        # Parameters for open-set (used in 4.2(c); 
        self.max_distance = float(max_distance)
        self.min_prob = float(min_prob)

        # The underlying gallery: class labels and embeddings (COLOR + GRAYSCALE).
        self.labels: list[str] = []
        self.embeddings = np.empty((0, FaceNet.embedding_dimensionality), dtype=np.float32)       # color
        self.embeddings_gray = np.empty((0, FaceNet.embedding_dimensionality), dtype=np.float32)  # grayscale

        # Load face recognizer from pickle file if available.
        if os.path.exists(Config.REC_GALLERY):
            self.load()
# Save the trained model as a pickle file.
    def save(self):
        print("FaceRecognizer saving: {}".format(Config.REC_GALLERY))
        with open(Config.REC_GALLERY, "wb") as f:
            # Save both embedding sets for the dual-embedding extension.
            pickle.dump((self.labels, self.embeddings, self.embeddings_gray), f)
    # Load trained model from a pickle file.
    def load(self):
        print("FaceRecognizer loading: {}".format(Config.REC_GALLERY))
        with open(Config.REC_GALLERY, "rb") as f:
            payload = pickle.load(f)

        # Backward-compatible load (older runs may have stored only (labels, embeddings)).
        if isinstance(payload, tuple) and len(payload) == 2:
            self.labels, self.embeddings = payload
            self.embeddings = np.asarray(self.embeddings, dtype=np.float32)

            # If no grayscale gallery exists, fall back to using the color embeddings.
            self.embeddings_gray = np.asarray(self.embeddings, dtype=np.float32)

        elif isinstance(payload, tuple) and len(payload) == 3:
            self.labels, self.embeddings, self.embeddings_gray = payload
            self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
            self.embeddings_gray = np.asarray(self.embeddings_gray, dtype=np.float32)

        else:
            raise ValueError("Unexpected FaceRecognizer pickle format.")

    def _to_gray_bgr(self, face_bgr: np.ndarray) -> np.ndarray:
        """Convert BGR face to grayscale and back to 3-channel BGR (FaceNet expects 3 channels)."""
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Train face identification with a new face with labeled identity.
    def partial_fit(self, face, label):
        """
        Store two embeddings per sample:
          - embedding from the color (BGR) face image
          - embedding from the grayscale version (converted back to 3-channel)
        """
        if face is None or face.size == 0:
            return None

        label_str = str(label)

        emb_color = self.facenet.predict(face)
        emb_gray = self.facenet.predict(self._to_gray_bgr(face))

        # Append to gallery
        self.labels.append(label_str)
        self.embeddings = np.vstack([self.embeddings, emb_color[None, :]])
        self.embeddings_gray = np.vstack([self.embeddings_gray, emb_gray[None, :]])

        return None

    # Predict the identity for a new face.
    def predict(self, face) -> tuple[str, float, float]:
      
        # No gallery -> cannot predict.
        n = len(self.labels)
        if n == 0:
            return ("unknown", 0.0, float("inf"))

        # Compute query embeddings (color + grayscale)
        q_color = self.facenet.predict(face)
        q_gray = self.facenet.predict(self._to_gray_bgr(face))

        # Distances to all samples (Euclidean)
        # shapes: (n,)
        dist_color = np.linalg.norm(self.embeddings - q_color[None, :], axis=1)
        dist_gray = np.linalg.norm(self.embeddings_gray - q_gray[None, :], axis=1)

        # Combine distances from color and grayscale embeddings
        dist = 0.5 * (dist_color + dist_gray)
        # dist = np.minimum(dist_color, dist_gray)

        # Effective k
        k = min(self.num_neighbours, n)
        if k <= 0:
            return ("unknown", 0.0, float("inf"))

        # Indices of k smallest distances
        nn_idx = np.argpartition(dist, k - 1)[:k]
        nn_idx = nn_idx[np.argsort(dist[nn_idx])]  # sort for stable behavior

        nn_labels = [self.labels[i] for i in nn_idx]

        # Majority vote
        unique_labels, counts = np.unique(np.array(nn_labels, dtype=object), return_counts=True)
        max_count = int(np.max(counts))
        candidates = unique_labels[counts == max_count]

        # Tie-break: choose label whose best (minimum) neighbor distance is smallest.
        if len(candidates) == 1:
            pred_label = str(candidates[0])
        else:
            best_label = None
            best_score = float("inf")
            for c in candidates:
                c = str(c)
                c_dists = [dist[i] for i in nn_idx if self.labels[i] == c]
                c_score = float(np.min(c_dists)) if c_dists else float("inf")
                if c_score < best_score:
                    best_score = c_score
                    best_label = c
            pred_label = best_label if best_label is not None else str(candidates[0])

        # Posterior probability p(Ci|x) = ki / k for the predicted class
        ki = sum(1 for lab in nn_labels if lab == pred_label)
        prob = float(ki) / float(k)

        # Distance to predicted class: min distance among neighbors belonging to predicted class
        pred_neighbor_dists = [dist[i] for i in nn_idx if self.labels[i] == pred_label]
        dist_to_prediction = float(np.min(pred_neighbor_dists)) if pred_neighbor_dists else float("inf")

        # # --- Open-set decision rule (Exercise 4.2(c)) ---
        if (dist_to_prediction > self.max_distance) or (prob < self.min_prob):
            return ("unknown", prob, dist_to_prediction)
        
    

import os
import pickle
import numpy as np

from cvproj_exc.config import Config

# assumes FaceNet exists above (as in your file)
# from cvproj_exc.face_recognition import FaceNet  # not needed if in same file


class FaceClustering:
    # 25 not good work
    # 50 only 1 cluster error
    def __init__(self, num_clusters=5, max_iter=50):
        self.facenet = FaceNet()

        # Unlabeled gallery (embeddings only)
        self.embeddings = np.empty((0, FaceNet.embedding_dimensionality), dtype=np.float32)

        # k-means parameters
        self.num_clusters = int(num_clusters)
        self.max_iter = int(max_iter)

        # Results
        self.cluster_center = np.empty((self.num_clusters, FaceNet.embedding_dimensionality), dtype=np.float32)
        self.cluster_membership = []  # list[int]

        # Track objective (J) over iterations
        self.objective_history: list[float] = []

        # Optional: store convergence reason (nice for analysis)
        self.convergence_reason: str = ""

        # Load from pickle if available
        if os.path.exists(Config.CLUSTER_GALLERY):
            self.load()

    def save(self):
        print("FaceClustering saving: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "wb") as f:
            pickle.dump(
                (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership),
                f,
            )

    def load(self):
        print("FaceClustering loading: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "rb") as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = pickle.load(f)

        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        self.cluster_center = np.asarray(self.cluster_center, dtype=np.float32)

        # objective_history is not persisted
        self.objective_history = []
        self.convergence_reason = ""

    def partial_fit(self, face):
        """Extract and store an embedding for a new face (no labels)."""
        if face is None or getattr(face, "size", 0) == 0:
            return None

        emb = self.facenet.predict(face).astype(np.float32)  # (128,)
        self.embeddings = np.vstack([self.embeddings, emb[None, :]])
        return None

    def fit(self, seed: int | None = None):
        """
        k-means clustering in embedding space.

        - Initialize centers randomly from data (controlled by `seed`).
        - Iterate assign/update.
        - Store cluster centers + membership.
        - Store objective J over iterations in self.objective_history.
        """
        X = self.embeddings
        n = X.shape[0]

        self.objective_history = []
        self.convergence_reason = ""

        if n == 0:
            self.cluster_membership = []
            return None

        k = int(self.num_clusters)
        if k < 2:
            raise ValueError("num_clusters must be >= 2 for k-means.")
        if n < k:
            # Not enough samples: fallback to k=n
            k = n
            self.num_clusters = k
            self.cluster_center = np.empty((k, FaceNet.embedding_dimensionality), dtype=np.float32)

        rng = np.random.default_rng(seed)

        # Random initialization: choose k distinct samples as initial centers.
        init_idx = rng.choice(n, size=k, replace=False)
        C = X[init_idx].copy()  # (k, d)

        membership = np.full(n, -1, dtype=np.int32)

        # Precompute X norms for fast squared distance computation
        X2 = np.sum(X * X, axis=1, keepdims=True)  # (n, 1)

        eps = 1e-6  # convergence threshold on center movement

        for it in range(self.max_iter):
            # ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x·c
            C2 = np.sum(C * C, axis=1, keepdims=True).T  # (1, k)
            dist2 = X2 + C2 - 2.0 * (X @ C.T)            # (n, k)

            new_membership = np.argmin(dist2, axis=1).astype(np.int32)

            # Objective J: sum of squared distances to assigned centers
            obj = float(np.sum(dist2[np.arange(n), new_membership]))
            self.objective_history.append(obj)

            # Stop if assignments do not change
            if it > 0 and np.array_equal(new_membership, membership):
                membership = new_membership
                self.convergence_reason = "assignments unchanged"
                break

            membership = new_membership

            # Update centers
            C_new = np.empty_like(C)
            for j in range(k):
                idx = np.where(membership == j)[0]
                if idx.size == 0:
                    # Empty cluster: reinitialize to random point
                    C_new[j] = X[rng.integers(0, n)]
                else:
                    C_new[j] = np.mean(X[idx], axis=0)

            # Stop if centers barely move
            center_shift = float(np.linalg.norm(C_new - C))
            C = C_new
            if center_shift < eps:
                self.convergence_reason = "centers converged"
                break
        else:
            self.convergence_reason = "reached max_iter"

        self.cluster_center = C.astype(np.float32)
        self.cluster_membership = membership.tolist()
        return None

    def predict(self, face) -> tuple[int, np.ndarray]:
        """
        Re-identify a face by nearest cluster center.
        Returns:
          - best cluster index (argmin distance)
          - distance vector to all centers
        """
        if face is None or getattr(face, "size", 0) == 0:
            return (0, np.array([], dtype=np.float32))

        if self.cluster_center is None or self.cluster_center.size == 0:
            return (0, np.array([], dtype=np.float32))

        x = self.facenet.predict(face).astype(np.float32)  # (128,)
        dists = np.linalg.norm(self.cluster_center - x[None, :], axis=1).astype(np.float32)
        pred = int(np.argmin(dists))
        return pred, dists

    # ------------------- Helpers for Exercise 4.3 analysis -------------------

    def print_objective_table(self, max_rows: int | None = None):
        """
        Prints a simple table: iteration -> objective J.
        Use this for the "diagram or table" requirement (table version).
        """
        if not self.objective_history:
            print("No objective history yet. Call fit() first.")
            return

        rows = self.objective_history
        if max_rows is not None:
            rows = rows[:max_rows]

        print("Iteration\tObjective J")
        for i, J in enumerate(rows):
            print(f"{i}\t\t{J:.6f}")
        print(f"Converged in {len(self.objective_history)} iterations ({self.convergence_reason}).")
        print(f"Final J = {self.objective_history[-1]:.6f}")

    def analyze_initialization_sensitivity(self, num_runs: int = 5):
        """
        Runs k-means multiple times with different seeds and prints final J.
        If final J changes across runs => sensitive to initialization.
        """
        if self.embeddings.shape[0] == 0:
            print("No embeddings available. Add data via partial_fit() or load() first.")
            return

        finals = []
        iters = []

        for s in range(num_runs):
            self.fit(seed=s)
            finals.append(self.objective_history[-1])
            iters.append(len(self.objective_history))

        print("Initialization sensitivity test")
        for i in range(num_runs):
            print(f"Run {i} (seed={i}): final J={finals[i]:.6f}, iters={iters[i]}, reason={self.convergence_reason}")

        print(f"Final J range: {min(finals):.6f} .. {max(finals):.6f}")
        print("If this range is large, clustering is sensitive to initialization.")


# class FaceClustering:
#     # 25 not good work
#     # 50 only 1 cluster error
#     def __init__(self, num_clusters=5, max_iter=50): # intial 200
#         self.facenet = FaceNet()

#         # The underlying gallery: embeddings without class labels.
#         # Unlabeled gallery
#         self.embeddings = np.empty((0, FaceNet.embedding_dimensionality), dtype=np.float32)

#           # Number of cluster centers for k-means clustering
#         self.num_clusters = int(num_clusters)
#         # Cluster centers.
#         self.cluster_center = np.empty((self.num_clusters, FaceNet.embedding_dimensionality), dtype=np.float32)
#         # Cluster index associated with the different samples.
#         self.cluster_membership = []  # list[int] of length n_samples
#         # Maximum number of iterations for k-means clustering.
#         self.max_iter = int(max_iter)

#         # Track objective over iterations for analysis/plotting
#         self.objective_history: list[float] = []

#         # Load face clustering from pickle file if available.
#         if os.path.exists(Config.CLUSTER_GALLERY):
#             self.load()

# # Save the trained model as a pickle file.
#     def save(self):
#         print("FaceClustering saving: {}".format(Config.CLUSTER_GALLERY))
#         with open(Config.CLUSTER_GALLERY, "wb") as f:
#             pickle.dump(
#                 (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership),
#                 f,
#             )
#      # Load trained model from a pickle file
#     def load(self):
#         print("FaceClustering loading: {}".format(Config.CLUSTER_GALLERY))
#         with open(Config.CLUSTER_GALLERY, "rb") as f:
#             (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = (
#                 pickle.load(f)
#             )
#         self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
#         self.cluster_center = np.asarray(self.cluster_center, dtype=np.float32)

#         # objective_history is not persisted (by design); recompute by calling fit() if needed
#         self.objective_history = []

#     def partial_fit(self, face):
#         """
#         4.3(a)(1): Extract and store an embedding for a new face (no labels).
#         """
#         if face is None or getattr(face, "size", 0) == 0:
#             return None

#         emb = self.facenet.predict(face).astype(np.float32)  # (128,)
#         self.embeddings = np.vstack([self.embeddings, emb[None, :]])
#         return None

#     def fit(self):
#         """
#         4.3(a)(2): k-means clustering in embedding space.
#           - Initialize centers by sampling k embeddings at random from data.
#           - Iterate assign/update.
#           - Store cluster centers and membership labels.
#         4.3(a)(3): Track objective function value over iterations (self.objective_history).
#         """
#         X = self.embeddings
#         n = X.shape[0]
#         if n == 0:
#             self.cluster_membership = []
#             self.objective_history = []
#             return None

#         k = int(self.num_clusters)
#         if k < 2:
#             raise ValueError("num_clusters must be >= 2 for k-means.")
#         if n < k:
#             # Not enough samples for requested k. Make it safe and proceed with k=n.
#             k = n
#             self.num_clusters = k
#             self.cluster_center = np.empty((k, FaceNet.embedding_dimensionality), dtype=np.float32)

#         rng = np.random.default_rng()

#         # Random initialization: choose k distinct samples as initial centers.
#         init_idx = rng.choice(n, size=k, replace=False)
#         C = X[init_idx].copy()  # (k, d)

#         # Membership array
#         membership = np.full(n, -1, dtype=np.int32)

#         # Track objective values
#         self.objective_history = []

#         # Precompute X norms for fast distance computation
#         X2 = np.sum(X * X, axis=1, keepdims=True)  # (n, 1)

#         eps = 1e-6  # convergence threshold on center movement

#         for it in range(self.max_iter):
#             # Compute squared distances efficiently:
#             # ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x·c
#             C2 = np.sum(C * C, axis=1, keepdims=True).T  # (1, k)
#             dist2 = X2 + C2 - 2.0 * (X @ C.T)            # (n, k)

#             new_membership = np.argmin(dist2, axis=1).astype(np.int32)

#             # Objective: sum of squared distances to assigned centers
#             obj = float(np.sum(dist2[np.arange(n), new_membership]))
#             self.objective_history.append(obj)

#             # Stop if assignments do not change
#             if it > 0 and np.array_equal(new_membership, membership):
#                 membership = new_membership
#                 break

#             membership = new_membership

#             # Update centers
#             C_new = np.empty_like(C)
#             for j in range(k):
#                 idx = np.where(membership == j)[0]
#                 if idx.size == 0:
#                     # Empty cluster: reinitialize its center to a random data point
#                     C_new[j] = X[rng.integers(0, n)]
#                 else:
#                     C_new[j] = np.mean(X[idx], axis=0)

#             # Stop if centers barely move
#             center_shift = float(np.linalg.norm(C_new - C))
#             C = C_new
#             if center_shift < eps:
#                 break

#         self.cluster_center = C.astype(np.float32)
#         self.cluster_membership = membership.tolist()
#         return None

#     def predict(self, face) -> tuple[int, np.ndarray]:
#         """
#         4.3 re-identification interface:
#           - Return distances d_x = (d(c1,x),...,d(ck,x))^T
#           - Return best cluster index argmin_i d(ci, x)
#         """
#         if face is None or getattr(face, "size", 0) == 0:
#             return (0, np.array([], dtype=np.float32))

#         if self.cluster_center is None or self.cluster_center.size == 0:
#             return (0, np.array([], dtype=np.float32))

#         x = self.facenet.predict(face).astype(np.float32)  # (128,)
#         dists = np.linalg.norm(self.cluster_center - x[None, :], axis=1).astype(np.float32)
#         pred = int(np.argmin(dists))
#         return pred, dists
    

