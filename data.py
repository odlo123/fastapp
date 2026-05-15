"""
data.py — Static seed data for the hybrid recommender.
10 users · 30 products · Rwanda e-commerce context.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# 30 PRODUCTS
# ──────────────────────────────────────────────
PRODUCTS: list[dict] = [
    # ── Electronics ──────────────────────────
    {"id": "P01", "name": "Samsung Galaxy A14",  "category": "electronics", "subcategory": "smartphones",  "price": 120_000, "brand": "Samsung",  "tags": "phone android mobile camera"},
    {"id": "P02", "name": "Tecno Spark 10",       "category": "electronics", "subcategory": "smartphones",  "price": 85_000,  "brand": "Tecno",    "tags": "phone android mobile budget"},
    {"id": "P03", "name": "Infinix Hot 30",        "category": "electronics", "subcategory": "smartphones",  "price": 95_000,  "brand": "Infinix",  "tags": "phone android mobile gaming"},
    {"id": "P04", "name": "JBL Flip 6",            "category": "electronics", "subcategory": "audio",        "price": 65_000,  "brand": "JBL",      "tags": "speaker bluetooth portable music"},
    {"id": "P05", "name": "Anker PowerCore 20K",   "category": "electronics", "subcategory": "accessories",  "price": 22_000,  "brand": "Anker",    "tags": "powerbank charger portable battery"},
    {"id": "P06", "name": "Logitech MX Keys",      "category": "electronics", "subcategory": "accessories",  "price": 48_000,  "brand": "Logitech", "tags": "keyboard wireless productivity office"},
    {"id": "P07", "name": "HP 245 G9 Laptop",      "category": "electronics", "subcategory": "computers",    "price": 450_000, "brand": "HP",       "tags": "laptop computer work office student"},
    {"id": "P08", "name": "TP-Link WiFi Router",   "category": "electronics", "subcategory": "networking",   "price": 35_000,  "brand": "TP-Link",  "tags": "router wifi internet network home"},
    # ── Fashion ──────────────────────────────
    {"id": "P09", "name": "Kitenge Dress (Women)",  "category": "fashion",     "subcategory": "clothing",     "price": 18_000,  "brand": "Local",    "tags": "dress african fashion women kitenge"},
    {"id": "P10", "name": "Men's Casual Shirt",     "category": "fashion",     "subcategory": "clothing",     "price": 12_000,  "brand": "Local",    "tags": "shirt men casual office fashion"},
    {"id": "P11", "name": "Nike Air Max 270",       "category": "fashion",     "subcategory": "footwear",     "price": 95_000,  "brand": "Nike",     "tags": "shoes sneakers sport running"},
    {"id": "P12", "name": "Adidas Backpack",        "category": "fashion",     "subcategory": "bags",         "price": 40_000,  "brand": "Adidas",   "tags": "backpack bag school sport travel"},
    {"id": "P13", "name": "Leather Handbag",        "category": "fashion",     "subcategory": "bags",         "price": 35_000,  "brand": "Local",    "tags": "handbag leather women fashion accessories"},
    # ── Home & Living ─────────────────────────
    {"id": "P14", "name": "Nonstick Cookware Set",  "category": "home",        "subcategory": "kitchen",      "price": 28_000,  "brand": "Tefal",    "tags": "cooking kitchen pots nonstick home"},
    {"id": "P15", "name": "Electric Kettle 1.7L",   "category": "home",        "subcategory": "kitchen",      "price": 15_000,  "brand": "Ramtons",  "tags": "kettle kitchen electric water home"},
    {"id": "P16", "name": "Solar Table Lamp",        "category": "home",        "subcategory": "lighting",     "price": 8_000,   "brand": "Local",    "tags": "lamp solar light home energy saving"},
    {"id": "P17", "name": "Bed Sheet Set (King)",   "category": "home",        "subcategory": "bedding",      "price": 22_000,  "brand": "Cotton+",  "tags": "bedsheet bedroom sleep comfort home"},
    {"id": "P18", "name": "Wall Clock Modern",       "category": "home",        "subcategory": "decor",        "price": 10_000,  "brand": "Generic",  "tags": "clock decor wall home office"},
    # ── Food & Grocery ────────────────────────
    {"id": "P19", "name": "Arabica Coffee Beans",   "category": "food",        "subcategory": "beverages",    "price": 4_500,   "brand": "Rwanda Coop","tags": "coffee arabica rwandan beverage drink"},
    {"id": "P20", "name": "Honey Raw 500g",          "category": "food",        "subcategory": "condiments",   "price": 3_500,   "brand": "BeeFarm",  "tags": "honey natural organic food health"},
    {"id": "P21", "name": "Umuceli Rice 5kg",        "category": "food",        "subcategory": "staples",      "price": 5_000,   "brand": "MINIMEX",  "tags": "rice staple food cooking grocery"},
    {"id": "P22", "name": "Indomie Noodles 40pk",   "category": "food",        "subcategory": "staples",      "price": 12_000,  "brand": "Indomie",  "tags": "noodles instant food quick meal"},
    # ── Health & Beauty ───────────────────────
    {"id": "P23", "name": "Nivea Body Lotion 400ml","category": "beauty",      "subcategory": "skincare",     "price": 8_000,   "brand": "Nivea",    "tags": "lotion skincare body beauty moisturizer"},
    {"id": "P24", "name": "Shea Butter Natural",    "category": "beauty",      "subcategory": "skincare",     "price": 5_000,   "brand": "Local",    "tags": "shea butter natural skincare hair beauty"},
    {"id": "P25", "name": "Vitamin C Serum",         "category": "beauty",      "subcategory": "skincare",     "price": 15_000,  "brand": "Ordinary", "tags": "serum vitamin skincare face beauty glow"},
    {"id": "P26", "name": "Oral-B Toothbrush Pro",  "category": "beauty",      "subcategory": "oral care",    "price": 6_000,   "brand": "Oral-B",   "tags": "toothbrush oral hygiene health"},
    # ── Sports & Outdoor ─────────────────────
    {"id": "P27", "name": "Yoga Mat 6mm",            "category": "sports",      "subcategory": "fitness",      "price": 12_000,  "brand": "Generic",  "tags": "yoga mat fitness exercise sport health"},
    {"id": "P28", "name": "Resistance Bands Set",    "category": "sports",      "subcategory": "fitness",      "price": 9_000,   "brand": "Generic",  "tags": "bands fitness exercise sport resistance"},
    {"id": "P29", "name": "Football (Size 5)",       "category": "sports",      "subcategory": "team sports",  "price": 18_000,  "brand": "Adidas",   "tags": "football soccer ball sport team outdoor"},
    {"id": "P30", "name": "Hiking Water Bottle",     "category": "sports",      "subcategory": "outdoor",      "price": 7_000,   "brand": "Nalgene",  "tags": "bottle water hiking outdoor sport travel"},
]

# ──────────────────────────────────────────────
# 10 USERS
# ──────────────────────────────────────────────
USERS: list[dict] = [
    {"id": "U01", "name": "Amina",   "age_group": "18-25", "location": "Kigali"},
    {"id": "U02", "name": "Bosco",   "age_group": "26-35", "location": "Kigali"},
    {"id": "U03", "name": "Clarisse","age_group": "26-35", "location": "Musanze"},
    {"id": "U04", "name": "David",   "age_group": "18-25", "location": "Rubavu"},
    {"id": "U05", "name": "Espoir",  "age_group": "36-45", "location": "Huye"},
    {"id": "U06", "name": "Fiston",  "age_group": "18-25", "location": "Kigali"},
    {"id": "U07", "name": "Grace",   "age_group": "26-35", "location": "Kigali"},
    {"id": "U08", "name": "Hirwa",   "age_group": "36-45", "location": "Nyagatare"},
    {"id": "U09", "name": "Ineza",   "age_group": "18-25", "location": "Kigali"},
    {"id": "U10", "name": "Jules",   "age_group": "26-35", "location": "Rwamagana"},
]

# ──────────────────────────────────────────────
# USER-ITEM INTERACTION MATRIX
# Ratings: 1–5 (0 = not interacted)
# ──────────────────────────────────────────────
#          P01  P02  P03  P04  P05  P06  P07  P08  P09  P10  P11  P12  P13  P14  P15  P16  P17  P18  P19  P20  P21  P22  P23  P24  P25  P26  P27  P28  P29  P30
RAW_MATRIX = np.array([
    # U01 Amina   – young, beauty + fashion focused
    [0,   0,   0,   0,   4,   0,   0,   0,   5,   0,   4,   0,   5,   0,   3,   0,   4,   0,   0,   4,   0,   0,   5,   5,   4,   3,   0,   0,   0,   0],
    # U02 Bosco   – tech + sports
    [5,   4,   0,   3,   5,   4,   5,   3,   0,   3,   0,   4,   0,   0,   0,   0,   0,   0,   3,   0,   0,   0,   0,   0,   0,   0,   4,   4,   5,   3],
    # U03 Clarisse– home + food
    [0,   0,   0,   0,   0,   0,   0,   0,   3,   0,   0,   0,   4,   5,   4,   3,   5,   4,   4,   5,   5,   4,   0,   4,   0,   3,   3,   0,   0,   0],
    # U04 David   – budget tech + sports
    [0,   5,   4,   0,   3,   0,   0,   4,   0,   3,   4,   4,   0,   0,   0,   0,   0,   0,   0,   0,   4,   5,   0,   0,   0,   0,   5,   4,   5,   4],
    # U05 Espoir  – mixed, family man
    [0,   0,   0,   0,   0,   3,   4,   3,   0,   4,   0,   0,   0,   5,   4,   5,   5,   3,   5,   4,   5,   4,   0,   0,   0,   4,   0,   0,   0,   3],
    # U06 Fiston  – gamer + tech
    [4,   5,   5,   4,   4,   0,   0,   4,   0,   0,   5,   5,   0,   0,   0,   0,   0,   0,   0,   0,   3,   3,   0,   0,   0,   0,   4,   5,   5,   5],
    # U07 Grace   – beauty + home + fashion
    [0,   0,   0,   0,   0,   0,   0,   0,   5,   0,   4,   0,   5,   4,   4,   4,   5,   4,   4,   5,   0,   0,   5,   5,   5,   4,   3,   0,   0,   0],
    # U08 Hirwa   – business, productivity
    [3,   0,   0,   0,   4,   5,   5,   4,   0,   5,   0,   4,   0,   0,   3,   0,   0,   4,   4,   0,   3,   0,   0,   0,   0,   3,   0,   0,   0,   3],
    # U09 Ineza   – young woman, fashion + beauty + fitness
    [0,   3,   0,   0,   0,   0,   0,   0,   5,   0,   5,   5,   4,   0,   0,   0,   3,   0,   0,   3,   0,   0,   5,   4,   5,   4,   5,   5,   0,   4],
    # U10 Jules   – all-rounder
    [3,   0,   0,   4,   3,   3,   3,   0,   0,   3,   0,   3,   0,   3,   4,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3],
], dtype=np.float32)


def get_products_df() -> pd.DataFrame:
    return pd.DataFrame(PRODUCTS).set_index("id")

def get_users_df() -> pd.DataFrame:
    return pd.DataFrame(USERS).set_index("id")

def get_interaction_matrix() -> pd.DataFrame:
    product_ids = [p["id"] for p in PRODUCTS]
    user_ids    = [u["id"] for u in USERS]
    return pd.DataFrame(RAW_MATRIX, index=user_ids, columns=product_ids)
