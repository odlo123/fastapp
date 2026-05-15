"""
hybrid.py — Hybrid Recommendation Engine

Scoring formula (per candidate product):
─────────────────────────────────────────
  hybrid_score = α × cbf_score
               + β × cf_score
               + γ × popularity_score
               + δ × diversity_penalty

  α = 0.60  Content-Based  (product attributes)
  β = 0.40  Collaborative  (what similar users liked)

  Additional signals layered on top:
  γ = 0.10  Popularity boost (normalised interaction count)
  δ = 0.05  Intra-list diversity penalty (avoid same-category saturation)

Cold-start handling
───────────────────
  • New user  (0 interactions) → 100 % popularity + content similarity to
    best-selling items.
  • New item  (0 interactions) → CF score = 0, weight shifts to CBF.
  • Warm user (1-2 items)      → α=0.80, β=0.20 (trust content more).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

from content_based  import ContentBasedRecommender
from collaborative  import CollaborativeRecommender


@dataclass
class RecommendationResult:
    product_id:       str
    product_name:     str
    category:         str
    price:            int
    cbf_score:        float
    cf_score:         float
    popularity_score: float
    hybrid_score:     float
    reason:           str   # human-readable explanation


class HybridRecommender:
    # Default weights
    CBF_WEIGHT        = 0.60
    CF_WEIGHT         = 0.40
    POPULARITY_WEIGHT = 0.10
    DIVERSITY_PENALTY = 0.05

    # Warm-user threshold (number of rated items)
    WARM_THRESHOLD = 3

    def __init__(
        self,
        products_df:         pd.DataFrame,
        users_df:            pd.DataFrame,
        interaction_matrix:  pd.DataFrame,
    ):
        self.products    = products_df
        self.users       = users_df
        self.interactions = interaction_matrix

        self.cbf = ContentBasedRecommender(products_df)
        self.cf  = CollaborativeRecommender(interaction_matrix)

        # Pre-compute popularity scores (normalised)
        self._popularity = self._compute_popularity()

    # ─────────────────────────────────────────
    # Setup helpers
    # ─────────────────────────────────────────
    def _compute_popularity(self) -> dict[str, float]:
        """
        Popularity = weighted interaction score per product.
        Score = (number of non-zero ratings) * (mean of those ratings) / max_possible
        Normalised to [0, 1].
        """
        R = self.interactions.values                # (10, 30)
        nonzero_counts = (R > 0).sum(axis=0)       # (30,)
        mean_ratings   = np.where(
            nonzero_counts > 0,
            R.sum(axis=0) / np.maximum(nonzero_counts, 1),
            0.0,
        )
        raw_pop  = nonzero_counts * mean_ratings    # interaction-weighted
        max_pop  = raw_pop.max()
        pop_norm = raw_pop / max_pop if max_pop > 0 else raw_pop

        return {
            pid: float(pop_norm[i])
            for i, pid in enumerate(self.interactions.columns)
        }

    # ─────────────────────────────────────────
    # Adaptive weights
    # ─────────────────────────────────────────
    def _get_weights(self, n_rated: int) -> tuple[float, float]:
        """
        Adapt CBF/CF weights based on how much we know about the user.
          Cold  (0 rated) → 1.0 / 0.0  (pure content + popularity)
          Warm  (1-2)     → 0.80 / 0.20
          Normal(3+)      → 0.60 / 0.40
        """
        if n_rated == 0:
            return 1.0, 0.0
        elif n_rated < self.WARM_THRESHOLD:
            return 0.80, 0.20
        return self.CBF_WEIGHT, self.CF_WEIGHT

    # ─────────────────────────────────────────
    # Core recommendation
    # ─────────────────────────────────────────
    def recommend(
        self,
        user_id:        str,
        top_n:          int = 5,
        exclude_seen:   bool = True,
        category_filter: str | None = None,
    ) -> list[RecommendationResult]:
        """
        Generate top-N hybrid recommendations for a user.

        Parameters
        ----------
        user_id        : str   e.g. "U04"
        top_n          : int   number of items to return
        exclude_seen   : bool  hide already-rated products
        category_filter: str   optional category filter ("electronics", etc.)

        Returns
        -------
        list[RecommendationResult] sorted by hybrid_score descending
        """
        if user_id not in self.interactions.index:
            raise ValueError(f"Unknown user_id: {user_id!r}")

        user_row  = self.interactions.loc[user_id]
        n_rated   = int((user_row > 0).sum())
        α, β      = self._get_weights(n_rated)

        # ── Get component scores ──────────────
        cbf_scores = self.cbf.score_for_user(user_row, exclude_seen=exclude_seen)
        cf_scores  = self.cf.score_for_user(user_id,   exclude_seen=exclude_seen)

        # ── Combine ───────────────────────────
        candidates: list[RecommendationResult] = []
        for pid in self.interactions.columns:
            product = self.products.loc[pid]

            # Category filter
            if category_filter and product["category"] != category_filter:
                continue

            cbf_s = cbf_scores.get(pid, 0.0)
            cf_s  = cf_scores.get(pid,  0.0)
            pop_s = self._popularity.get(pid, 0.0)

            raw_hybrid = α * cbf_s + β * cf_s + self.POPULARITY_WEIGHT * pop_s

            # Build reason string
            reason = self._build_reason(α, β, cbf_s, cf_s, pop_s, n_rated, product)

            candidates.append(RecommendationResult(
                product_id       = pid,
                product_name     = product["name"],
                category         = product["category"],
                price            = int(product["price"]),
                cbf_score        = round(cbf_s, 4),
                cf_score         = round(cf_s,  4),
                popularity_score = round(pop_s, 4),
                hybrid_score     = round(raw_hybrid, 4),
                reason           = reason,
            ))

        # ── Diversity re-ranking ──────────────
        candidates = self._diversify(candidates)

        # Sort and return top-N
        candidates.sort(key=lambda r: r.hybrid_score, reverse=True)
        return candidates[:top_n]

    # ─────────────────────────────────────────
    # Diversity re-ranking
    # ─────────────────────────────────────────
    def _diversify(self, candidates: list[RecommendationResult]) -> list[RecommendationResult]:
        """
        Apply a soft intra-list diversity penalty:
        if a category already has 2+ candidates in the top slice,
        slightly reduce subsequent same-category items.
        """
        category_counts: dict[str, int] = {}
        for r in candidates:
            count = category_counts.get(r.category, 0)
            if count >= 2:
                penalty_factor = 1.0 - self.DIVERSITY_PENALTY * (count - 1)
                r.hybrid_score = round(r.hybrid_score * max(penalty_factor, 0.5), 4)
            category_counts[r.category] = count + 1
        return candidates

    # ─────────────────────────────────────────
    # Explanation builder
    # ─────────────────────────────────────────
    def _build_reason(
        self,
        α: float, β: float,
        cbf_s: float, cf_s: float, pop_s: float,
        n_rated: int,
        product: pd.Series,
    ) -> str:
        parts = []
        if n_rated == 0:
            parts.append("Popular item for new users")
        else:
            if cbf_s > 0.5:
                parts.append(f"Matches your taste in {product['category']}")
            if cf_s > 0.5:
                parts.append("People like you also liked this")
            if pop_s > 0.7:
                parts.append("Trending in Rwanda")
        return "; ".join(parts) if parts else "Recommended for you"

    # ─────────────────────────────────────────
    # Live update
    # ─────────────────────────────────────────
    def record_interaction(
        self,
        user_id: str,
        product_id: str,
        rating: float,
    ) -> None:
        """
        Record a new user interaction and update models in real-time.
        Both the in-memory interaction matrix and the CF model are updated.
        """
        if user_id not in self.interactions.index:
            raise ValueError(f"Unknown user: {user_id!r}")
        if product_id not in self.interactions.columns:
            raise ValueError(f"Unknown product: {product_id!r}")

        # Update interaction matrix
        self.interactions.loc[user_id, product_id] = float(rating)

        # Update CF (refits internally)
        self.cf.update_interaction(user_id, product_id, rating)

        # Refresh popularity scores
        self._popularity = self._compute_popularity()
