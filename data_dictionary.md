# Nairobi House Price Prediction — Data Dictionary

**File:** `data/raw_listings.csv`  
**Format:** CSV (comma-separated values)  
**Total Rows:** ~4,734  
**Total Columns:** 9  
**Target Column:** `Price_KES`  
**Coverage:** 30+ Nairobi suburbs  
**Date Range:** Nov 2024 – Feb 2026  

---

## Column Definitions

| Column Name | Data Type | Description | Possible Values / Range | Nullable |
|---|---|---|---|---|
| `Location` | string | Nairobi suburb or neighbourhood where the property is located | Westlands, Kilimani, Karen, Lavington, etc. | No |
| `Property_Type` | string | Category of the property | Apartment, House, Townhouse, Studio, Bedsitter, Villa, Penthouse | No |
| `Bedrooms` | integer | Number of bedrooms in the property | 0 (studio/bedsitter), 1, 2, 3, 4, 5+ | No |
| `Bathrooms` | integer | Number of bathrooms in the property | 1, 2, 3, 4+ | No |
| `Size_SQM` | integer | Total floor area in square metres. 0 means not provided | 0 – 1000 | Yes (0 = unknown) |
| `Amenities` | string | Comma-separated list of available amenities | Security, Parking, Pool, Gym, Garden, Balcony, DSQ, Generator, Borehole, Lift, CCTV | No |
| `Price_KES` | integer | Listed price in Kenyan Shillings. 0 means not disclosed | 0 – 2,000,000,000 | Yes (0 = undisclosed) |
| `Listing_Type` | string | Whether the property is for sale or for rent | For Sale, For Rent | No |
| `Listing_Date` | date | Date the listing was scraped or recorded (YYYY-MM-DD) | 2024-11-01 to 2026-02-17 | No |

---

## Data Sources

| Source | Approx. Rows | Notes |
|---|---|---|
| `property24.co.ke` | ~960 | Scraped via scrape_multi.py |
| `nairobi_properties_clean.csv` | ~50 | Kaggle — Property24 scrape (May 2025) |
| `houses-for-sale.csv` | ~1,876 | Kaggle — BuyRentKenya sale listings |
| `rent_apts.csv` | ~1,848 | Kaggle — BuyRentKenya rental apartments |

---

## Notes for ML Use

| Column | ML Consideration |
|---|---|
| `Price_KES` | Target variable. Drop rows where Price_KES = 0 before training |
| `Size_SQM` | Size_SQM = 0 means unknown — impute with median per bedroom count |
| `Location` | High-cardinality categorical (30+ values) — use target encoding or embeddings |
| `Amenities` | Multi-label string — one-hot encode each amenity as a separate binary feature |
| `Listing_Type` | Split For Sale and For Rent into separate models — price scales differ by ~100x |
| `Listing_Date` | Drop or extract month/year features for seasonality analysis |