"""
serve.py — FastAPI Real-Time Serving Layer

Endpoints
─────────
  GET  /health                          Liveness check
  GET  /users                           List all users
  GET  /products                        List all products
  GET  /recommend/{user_id}             Get top-N recommendations
  POST /interact                        Log a user interaction (updates model)
  GET  /similar-items/{product_id}      Content-based similar products
  GET  /similar-users/{user_id}         Collaborative similar users
  GET  /scores/{user_id}/{product_id}   Debug: component scores for one pair

Run:
  uvicorn serve:app --reload --port 8000
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Internal imports ──────────────────────────────────────────────────────────
from data       import get_products_df, get_users_df, get_interaction_matrix
from hybrid     import HybridRecommender, RecommendationResult
from shareapi import router as share_router


app.include_router(
    share_router
)
# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Rwanda E-Commerce Hybrid Recommender",
    description = "Content-Based (60%) + Collaborative Filtering (40%) real-time API",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Model initialisation (singleton) ──────────────────────────────────────────
print("⚙️  Initialising recommendation models …")
_t0      = time.perf_counter()
products = get_products_df()
users    = get_users_df()
matrix   = get_interaction_matrix()
engine   = HybridRecommender(products, users, matrix)
_elapsed = time.perf_counter() - _t0
print(f"✅  Models ready in {_elapsed * 1000:.1f} ms")


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class RecommendationItem(BaseModel):
    rank:             int
    product_id:       str
    product_name:     str
    category:         str
    price_rwf:        int
    cbf_score:        float = Field(..., description="Content-based score [0,1]")
    cf_score:         float = Field(..., description="Collaborative score [0,1]")
    popularity_score: float = Field(..., description="Popularity score [0,1]")
    hybrid_score:     float = Field(..., description="Final hybrid score [0,1]")
    reason:           str

class RecommendResponse(BaseModel):
    user_id:       str
    user_name:     str
    top_n:         int
    cbf_weight:    float = 0.60
    cf_weight:     float = 0.40
    generated_at:  str
    latency_ms:    float
    recommendations: list[RecommendationItem]

class InteractRequest(BaseModel):
    user_id:    str = Field(..., example="U01")
    product_id: str = Field(..., example="P05")
    rating:     float = Field(..., ge=1.0, le=5.0, example=4.5)

    @field_validator("rating")
    @classmethod
    def round_rating(cls, v: float) -> float:
        return round(v * 2) / 2   # snap to 0.5 steps

class InteractResponse(BaseModel):
    status:     str
    user_id:    str
    product_id: str
    rating:     float
    message:    str

class SimilarItemResponse(BaseModel):
    query_product_id:   str
    query_product_name: str
    similar_products:   list[dict]

class SimilarUserResponse(BaseModel):
    query_user_id:   str
    query_user_name: str
    similar_users:   list[dict]

class ScoreDebugResponse(BaseModel):
    user_id:          str
    product_id:       str
    cbf_score:        float
    cf_score:         float
    popularity_score: float
    hybrid_score:     float


# ── Helper ────────────────────────────────────────────────────────────────────
def _result_to_item(rank: int, r: RecommendationResult) -> RecommendationItem:
    return RecommendationItem(
        rank             = rank,
        product_id       = r.product_id,
        product_name     = r.product_name,
        category         = r.category,
        price_rwf        = r.price,
        cbf_score        = r.cbf_score,
        cf_score         = r.cf_score,
        popularity_score = r.popularity_score,
        hybrid_score     = r.hybrid_score,
        reason           = r.reason,
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {
        "status":    "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "model":     "hybrid-cbf60-cf40",
        "n_users":   len(users),
        "n_products": len(products),
    }


@app.get("/users", tags=["Data"])
def list_users():
    return users.reset_index().to_dict(orient="records")


@app.get("/products", tags=["Data"])
def list_products(category: Optional[str] = Query(None)):
    df = products.reset_index()
    if category:
        df = df[df["category"] == category]
    return df.to_dict(orient="records")


@app.get(
    "/recommend/{user_id}",
    response_model = RecommendResponse,
    tags           = ["Recommendations"],
    summary        = "Get personalised recommendations for a user",
)
def recommend(
    user_id:         str,
    top_n:           int            = Query(5, ge=1, le=30),
    exclude_seen:    bool           = Query(True),
    category_filter: Optional[str] = Query(None, description="Filter by category"),
):
    if user_id not in users.index:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")

    t0      = time.perf_counter()
    results = engine.recommend(
        user_id        = user_id,
        top_n          = top_n,
        exclude_seen   = exclude_seen,
        category_filter= category_filter,
    )
    latency = (time.perf_counter() - t0) * 1000  # ms

    user_name = users.loc[user_id, "name"]
    items     = [_result_to_item(i + 1, r) for i, r in enumerate(results)]

    return RecommendResponse(
        user_id         = user_id,
        user_name       = user_name,
        top_n           = top_n,
        generated_at    = datetime.utcnow().isoformat(),
        latency_ms      = round(latency, 2),
        recommendations = items,
    )


@app.post(
    "/interact",
    response_model = InteractResponse,
    tags           = ["Interactions"],
    summary        = "Record a user interaction and update the model in real-time",
)
def record_interaction(body: InteractRequest):
    if body.user_id not in users.index:
        raise HTTPException(status_code=404, detail=f"User '{body.user_id}' not found.")
    if body.product_id not in products.index:
        raise HTTPException(status_code=404, detail=f"Product '{body.product_id}' not found.")

    engine.record_interaction(body.user_id, body.product_id, body.rating)

    return InteractResponse(
        status     = "success",
        user_id    = body.user_id,
        product_id = body.product_id,
        rating     = body.rating,
        message    = f"Interaction recorded. CF model updated in real-time.",
    )


@app.get(
    "/similar-items/{product_id}",
    response_model = SimilarItemResponse,
    tags           = ["Recommendations"],
    summary        = "Content-based similar products",
)
def similar_items(product_id: str, top_n: int = Query(5, ge=1, le=15)):
    if product_id not in products.index:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found.")

    similars = engine.cbf.similar_items(product_id, top_n=top_n)
    enriched = [
        {
            "product_id":   pid,
            "product_name": products.loc[pid, "name"],
            "category":     products.loc[pid, "category"],
            "price_rwf":    int(products.loc[pid, "price"]),
            "similarity":   round(sim, 4),
        }
        for pid, sim in similars
    ]

    return SimilarItemResponse(
        query_product_id   = product_id,
        query_product_name = products.loc[product_id, "name"],
        similar_products   = enriched,
    )


@app.get(
    "/similar-users/{user_id}",
    response_model = SimilarUserResponse,
    tags           = ["Recommendations"],
    summary        = "Collaborative similar users",
)
def similar_users(user_id: str, top_n: int = Query(3, ge=1, le=9)):
    if user_id not in users.index:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")

    similars = engine.cf.similar_users(user_id, top_n=top_n)
    enriched = [
        {
            "user_id":    uid,
            "user_name":  users.loc[uid, "name"],
            "similarity": round(sim, 4),
        }
        for uid, sim in similars
    ]

    return SimilarUserResponse(
        query_user_id   = user_id,
        query_user_name = users.loc[user_id, "name"],
        similar_users   = enriched,
    )


@app.get(
    "/scores/{user_id}/{product_id}",
    response_model = ScoreDebugResponse,
    tags           = ["Debug"],
    summary        = "Inspect component scores for a (user, product) pair",
)
def debug_scores(user_id: str, product_id: str):
    if user_id not in users.index:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")
    if product_id not in products.index:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found.")

    user_row   = engine.interactions.loc[user_id]
    cbf_scores = engine.cbf.score_for_user(user_row, exclude_seen=False)
    cf_scores  = engine.cf.score_for_user(user_id,   exclude_seen=False)
    pop        = engine.popularity if hasattr(engine, "popularity") else engine._popularity

    cbf_s  = cbf_scores.get(product_id, 0.0)
    cf_s   = cf_scores.get(product_id,  0.0)
    pop_s  = pop.get(product_id, 0.0)
    hybrid = 0.60 * cbf_s + 0.40 * cf_s + 0.10 * pop_s

    return ScoreDebugResponse(
        user_id          = user_id,
        product_id       = product_id,
        cbf_score        = round(cbf_s,  4),
        cf_score         = round(cf_s,   4),
        popularity_score = round(pop_s,  4),
        hybrid_score     = round(hybrid, 4),
    )
