"""
collaborative.py — Collaborative Filtering (CF) engine.

Two complementary strategies combined:

A) SVD Matrix Factorization (primary)
   - Decompose the (10 × 30) user-item rating matrix via truncated SVD.
   - Reconstruct predicted ratings for every (user, item) pair.
   - Gives a smooth, latent-factor–based prediction.

B) User-Based Cosine Similarity (secondary / fallback)
   - Find the K most similar users.
   - Weighted average of their ratings as the prediction.
   - Useful when SVD variance is low (cold-start or sparse rows).

Final CF score = 0.7 * SVD_score + 0.3 * UserKNN_score  (both normalised).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds


class CollaborativeRecommender:
    SVD_FACTORS   = 5    # latent factors (limited by min(10, 30))
    KNN_K         = 3    # similar-user neighbours
    SVD_WEIGHT    = 0.7
    KNN_WEIGHT    = 0.3

    def __init__(self, interaction_matrix: pd.DataFrame):
        """
        Parameters
        ----------
        interaction_matrix : pd.DataFrame
            shape (n_users, n_products), values 0–5, 0 = not rated.
        """
        self.matrix   = interaction_matrix           # raw
        self._user_ids    = interaction_matrix.index.tolist()
        self._product_ids = interaction_matrix.columns.tolist()
        self._uid_to_idx  = {u: i for i, u in enumerate(self._user_ids)}
        self._pid_to_idx  = {p: i for i, p in enumerate(self._product_ids)}

        self._R: np.ndarray = interaction_matrix.values.astype(np.float32)  # (10, 30)
        self._svd_predictions: np.ndarray   # (10, 30)
        self._user_sim_matrix: np.ndarray   # (10, 10)

        self._fit()

    # ─────────────────────────────────────────
    # Fitting
    # ─────────────────────────────────────────
    def _fit(self) -> None:
        # --- Mean-centre per user (treat 0 as missing) ---
        R_filled = self._R.copy()
        user_means = np.zeros(len(self._user_ids))
        for i in range(len(self._user_ids)):
            rated = R_filled[i, R_filled[i] > 0]
            mean  = rated.mean() if len(rated) > 0 else 3.0
            user_means[i] = mean
            R_filled[i, R_filled[i] == 0] = mean   # fill missing with mean

        R_centered = R_filled - user_means[:, np.newaxis]  # (10, 30)

        # --- Truncated SVD ---
        k = min(self.SVD_FACTORS, min(R_centered.shape) - 1)
        U, sigma, Vt = svds(R_centered, k=k)
        # Reconstruct and add back user means
        self._svd_predictions = (U @ np.diag(sigma) @ Vt) + user_means[:, np.newaxis]
        self._svd_predictions = np.clip(self._svd_predictions, 1.0, 5.0)

        # --- User-user cosine similarity (on filled matrix) ---
        self._user_sim_matrix = cosine_similarity(R_filled)  # (10, 10)

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────
    def score_for_user(
        self,
        user_id: str,
        exclude_seen: bool = True,
    ) -> dict[str, float]:
        """
        Return CF scores (normalised to [0,1]) for all products.

        Parameters
        ----------
        user_id      : str   e.g. "U03"
        exclude_seen : bool  zero-out products the user already rated

        Returns
        -------
        dict[product_id → cf_score ∈ [0,1]]
        """
        if user_id not in self._uid_to_idx:
            return {pid: 0.5 for pid in self._product_ids}

        u_idx = self._uid_to_idx[user_id]

        # A) SVD predicted ratings (1-5) → normalise to [0,1]
        svd_raw   = self._svd_predictions[u_idx]         # (30,)
        svd_norm  = (svd_raw - 1.0) / 4.0               # map [1,5]→[0,1]

        # B) User-KNN predictions
        knn_norm  = self._knn_predict(u_idx)             # (30,) already [0,1]

        # Blend
        cf_scores = self.SVD_WEIGHT * svd_norm + self.KNN_WEIGHT * knn_norm  # (30,)

        raw_ratings = self._R[u_idx]
        results: dict[str, float] = {}
        for pid, score in zip(self._product_ids, cf_scores):
            p_idx = self._pid_to_idx[pid]
            if exclude_seen and raw_ratings[p_idx] > 0:
                results[pid] = 0.0
            else:
                results[pid] = float(np.clip(score, 0.0, 1.0))

        return results

    def _knn_predict(self, u_idx: int) -> np.ndarray:
        """
        Weighted average of K most similar users' ratings (normalised).
        Returns shape (30,).
        """
        sims = self._user_sim_matrix[u_idx].copy()
        sims[u_idx] = -1.0                              # exclude self
        top_k = np.argsort(sims)[::-1][: self.KNN_K]

        weights = sims[top_k]
        weights = np.clip(weights, 0.0, 1.0)
        if weights.sum() < 1e-9:
            return np.full(len(self._product_ids), 0.5)

        neighbour_ratings = self._R[top_k]               # (K, 30)
        weighted = (neighbour_ratings * weights[:, np.newaxis]).sum(axis=0)
        pred     = weighted / weights.sum()              # (30,) scale 0-5
        return np.clip(pred / 5.0, 0.0, 1.0)

    def similar_users(self, user_id: str, top_n: int = 3) -> list[tuple[str, float]]:
        """Return the top-N most similar users."""
        u_idx = self._uid_to_idx[user_id]
        sims  = self._user_sim_matrix[u_idx].copy()
        sims[u_idx] = -1.0
        ranked = np.argsort(sims)[::-1]
        results = []
        for i in ranked[:top_n]:
            results.append((self._user_ids[i], float(sims[i])))
        return results

    def update_interaction(self, user_id: str, product_id: str, rating: float) -> None:
        """
        Live update: add / overwrite a rating, then refit.
        Called by the serving layer on every new interaction.
        """
        if user_id not in self._uid_to_idx or product_id not in self._pid_to_idx:
            return
        u_idx = self._uid_to_idx[user_id]
        p_idx = self._pid_to_idx[product_id]
        self._R[u_idx, p_idx] = float(np.clip(rating, 1.0, 5.0))
        self._fit()   # small matrix → refit is cheap (<1 ms)
