"""
Merge Script — Combines all 4 datasets into raw_listings.csv
=============================================================
Run:    python merge.py

Files expected in data/ folder:
    data/raw_listings.csv              ← your 960 scraped rows
    data/nairobi_properties_clean.csv  ← kaggle file 1  (~50 rows)
    data/houses-for-sale.csv           ← kaggle file 2  (~1,876 rows)
    data/rent_apts.csv                 ← kaggle file 3  (~1,848 rows)

Output:
    data/raw_listings.csv              ← all merged + shuffled
"""

import pandas as pd
import re
import os
import random
from datetime import datetime, timedelta

COLS = ["Location","Property_Type","Bedrooms","Bathrooms",
        "Size_SQM","Amenities","Price_KES","Listing_Type","Listing_Date"]

def rand_date():
    return (datetime.now() - timedelta(days=random.randint(0,90))).strftime("%Y-%m-%d")

def parse_beds(text):
    m = re.search(r"(\d+)[\s\-]?bed", str(text).lower())
    return int(m.group(1)) if m else 0

def parse_baths(text, beds=0):
    m = re.search(r"(\d+)[\s\-]?bath", str(text).lower())
    return int(m.group(1)) if m else max(1, beds)

def parse_price(text):
    t = str(text).replace(",","").replace(" ","")
    m = re.search(r"(\d+)", t)
    return int(m.group(1)) if m else 0

def classify(text):
    t = str(text).lower()
    for kw, label in [
        ("villa","Villa"),("townhouse","Townhouse"),("maisonette","House"),
        ("bungalow","House"),("studio","Studio"),("bedsitter","Bedsitter"),
        ("apartment","Apartment"),("flat","Apartment"),("house","House"),
    ]:
        if kw in t: return label
    return "Apartment"

def parse_size(text):
    m = re.search(r"(\d+)\s*m[²2]", str(text), re.I)
    if m: return int(m.group(1))
    m2 = re.search(r"(\d+)\s*sq", str(text), re.I)
    if m2: return int(int(m2.group(1)) * 0.0929)
    return 0

frames = []
print("="*50)
print("MERGING FILES")
print("="*50)

# ── FILE 1: raw_listings.csv (scraped) ───────────────────────
f1 = "data/raw_listings.csv"
if os.path.exists(f1):
    df1 = pd.read_csv(f1)
    frames.append(df1)
    print(f"  raw_listings.csv             : {len(df1):,} rows")
else:
    print(f"  WARNING: {f1} not found")

# ── FILE 2: nairobi_properties_clean.csv ─────────────────────
f2 = "data/nairobi_properties_clean.csv"
if os.path.exists(f2):
    df2   = pd.read_csv(f2)
    rows2 = []
    for _, r in df2.iterrows():
        price = int(r["price_numeric"]) if pd.notna(r.get("price_numeric")) and r["price_numeric"] < 2e9 else 0
        loc   = str(r.get("neighborhood", r.get("location","Nairobi")))
        rows2.append({
            "Location":      loc,
            "Property_Type": "Apartment",
            "Bedrooms":      0,
            "Bathrooms":     1,
            "Size_SQM":      0,
            "Amenities":     "Security, Parking",
            "Price_KES":     price,
            "Listing_Type":  "For Sale",
            "Listing_Date":  rand_date(),
        })
    frames.append(pd.DataFrame(rows2, columns=COLS))
    print(f"  nairobi_properties_clean.csv : {len(rows2):,} rows")
else:
    print(f"  WARNING: {f2} not found — copy it into data/ folder")

# ── FILE 3: houses-for-sale.csv ──────────────────────────────
f3 = "data/houses-for-sale.csv"
if os.path.exists(f3):
    df3   = pd.read_csv(f3)
    rows3 = []
    for _, r in df3.iterrows():
        title      = str(r.get("title",""))
        loc        = str(r.get("location","Nairobi")).split(",")[0].strip()
        size_text  = str(r.get("size",""))
        price_text = str(r.get("selling price",""))
        beds       = parse_beds(title + " " + size_text)
        baths      = parse_baths(title + " " + size_text, beds)
        rows3.append({
            "Location":      loc,
            "Property_Type": classify(title),
            "Bedrooms":      beds,
            "Bathrooms":     max(1, baths),
            "Size_SQM":      parse_size(size_text),
            "Amenities":     "Security, Parking",
            "Price_KES":     parse_price(price_text),
            "Listing_Type":  "For Sale",
            "Listing_Date":  rand_date(),
        })
    frames.append(pd.DataFrame(rows3, columns=COLS))
    print(f"  houses-for-sale.csv          : {len(rows3):,} rows")
else:
    print(f"  WARNING: {f3} not found — copy it into data/ folder")

# ── FILE 4: rent_apts.csv ────────────────────────────────────
f4 = "data/rent_apts.csv"
if os.path.exists(f4):
    df4   = pd.read_csv(f4)
    rows4 = []
    for _, r in df4.iterrows():
        loc   = str(r.get("Neighborhood","Nairobi")).split(",")[0].strip()
        beds  = int(r["Bedrooms"])  if pd.notna(r.get("Bedrooms"))  else 0
        baths = int(r["Bathrooms"]) if pd.notna(r.get("Bathrooms")) else max(1, beds)
        size  = int(r["sq_mtrs"])   if pd.notna(r.get("sq_mtrs")) and r["sq_mtrs"] > 0 else 0
        price = parse_price(r.get("Price", 0))
        rows4.append({
            "Location":      loc,
            "Property_Type": "Apartment",
            "Bedrooms":      max(0, beds),
            "Bathrooms":     max(1, baths),
            "Size_SQM":      size,
            "Amenities":     "Security, Parking",
            "Price_KES":     price,
            "Listing_Type":  "For Rent",
            "Listing_Date":  rand_date(),
        })
    frames.append(pd.DataFrame(rows4, columns=COLS))
    print(f"  rent_apts.csv                : {len(rows4):,} rows")
else:
    print(f"  WARNING: {f4} not found — copy it into data/ folder")

# ── MERGE + SHUFFLE ──────────────────────────────────────────
if frames:
    df_final = pd.concat(frames, ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    df_final.to_csv("data/raw_listings.csv", index=False)
    print(f"\n  Saved to data/raw_listings.csv")
    print(f"  Total rows: {len(df_final):,}")