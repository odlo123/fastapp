"""
content_based.py — Content-Based Filtering (CBF) engine.

Strategy
--------
1. Build a product feature vector from:
   - TF-IDF on (category + subcategory + brand + tags)   → sparse text similarity
   - Min-max normalised price                             → scalar feature
2. Compute pairwise cosine similarity → (30 × 30) matrix.
3. For a user: average the feature vectors of their liked items
   (rating ≥ threshold), then rank all other products by cosine
   similarity to that average profile vector.

Output: dict[product_id → cbf_score ∈ [0, 1]] for every product.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class ContentBasedRecommender:
    LIKE_THRESHOLD = 3.0          # ratings ≥ this count as "liked"
    PRICE_WEIGHT   = 0.2          # relative weight for the price feature

    def __init__(self, products_df: pd.DataFrame):
        self.products = products_df
        self._item_matrix: np.ndarray   # (n_products, n_features)
        self._product_ids: list[str]
        self._build_feature_matrix()

    # ─────────────────────────────────────────
    # Build feature matrix
    # ─────────────────────────────────────────
    def _build_feature_matrix(self) -> None:
        df = self.products.reset_index()           # bring "id" back as column
        self._product_ids = df["id"].tolist()

        # --- Text features ---
        text_corpus = (
            df["category"]    + " " +
            df["subcategory"] + " " +
            df["brand"]       + " " +
            df["tags"]
        )
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=200,
            sublinear_tf=True,
        )
        text_matrix = tfidf.fit_transform(text_corpus).toarray()  # (30, V)

        # --- Price feature ---
        price_vals = df["price"].values.reshape(-1, 1).astype(np.float32)
        scaler     = MinMaxScaler()
        price_norm = scaler.fit_transform(price_vals)              # (30, 1)

        # Weighted concatenation
        self._item_matrix = np.hstack([
            text_matrix,
            price_norm * self.PRICE_WEIGHT
        ]).astype(np.float32)                                      # (30, V+1)

        # Pre-compute full similarity matrix (30×30) for fast item-item lookup
        self._sim_matrix: np.ndarray = cosine_similarity(self._item_matrix)
        self._id_to_idx: dict[str, int] = {
            pid: i for i, pid in enumerate(self._product_ids)
        }

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────
    def score_for_user(
        self,
        interaction_row: pd.Series,
        exclude_seen: bool = True,
    ) -> dict[str, float]:
        """
        Given one user's interaction row (product_id → rating),
        return CBF scores for all products.

        Parameters
        ----------
        interaction_row : pd.Series  index=product_ids, values=ratings (0=unseen)
        exclude_seen    : bool       zero-out products already rated

        Returns
        -------
        dict[product_id → cbf_score] (all products, scores in [0,1])
        """
        liked_ids = [
            pid for pid, rating in interaction_row.items()
            if rating >= self.LIKE_THRESHOLD
        ]

        if not liked_ids:
            # Cold-start: return uniform scores
            return {pid: 0.5 for pid in self._product_ids}

        # Average feature vector of liked products
        liked_idxs   = [self._id_to_idx[pid] for pid in liked_ids]
        liked_matrix = self._item_matrix[liked_idxs]        # (k, V+1)
        user_profile = liked_matrix.mean(axis=0, keepdims=True)  # (1, V+1)

        # Cosine similarity of user profile to every product
        raw_scores = cosine_similarity(user_profile, self._item_matrix)[0]  # (30,)

        scores: dict[str, float] = {}
        for pid, score in zip(self._product_ids, raw_scores):
            if exclude_seen and interaction_row.get(pid, 0) >= self.LIKE_THRESHOLD:
                scores[pid] = 0.0
            else:
                scores[pid] = float(np.clip(score, 0.0, 1.0))

        return scores

    def similar_items(self, product_id: str, top_n: int = 5) -> list[tuple[str, float]]:
        """Return the top-N most similar products to a given product."""
        idx = self._id_to_idx[product_id]
        sims = self._sim_matrix[idx]
        ranked = np.argsort(sims)[::-1]
        results = []
        for i in ranked:
            pid = self._product_ids[i]
            if pid != product_id:
                results.append((pid, float(sims[i])))
            if len(results) >= top_n:
                break
        return results
