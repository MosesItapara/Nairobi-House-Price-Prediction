# Nairobi House Price Prediction

A machine learning project that predicts property prices in Nairobi, Kenya, using real estate data scraped from local property websites.

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

##  Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Key Insights](#key-insights)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [Contact](#contact)

##  Overview

This project aims to predict house prices in Nairobi based on features like location, property size, number of bedrooms/bathrooms, and available amenities. The project covers the full data science pipeline: data collection, cleaning, exploratory analysis, modeling, and deployment.

##  Features

- **Web Scraping:** Automated data collection from Property24.co.ke
- **Data Cleaning:** Comprehensive preprocessing and feature engineering
- **ML Models:** Comparison of Linear Regression, Random Forest, and XGBoost
- **Interactive Web App:** User-friendly Streamlit interface for price predictions
- **Explainability:** Clear insights into what drives property prices

##  Tech Stack

**Languages & Libraries:**
- Python 3.11
- pandas, NumPy (Data processing)
- scikit-learn, XGBoost (Machine learning)
- Streamlit (Web app)
- BeautifulSoup, requests (Web scraping)
- Matplotlib, Seaborn (Visualization)

**Deployment:**
- Streamlit Cloud / Render
- Git & GitHub

##  Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Git

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/nairobi-house-price-prediction.git
cd nairobi-house-price-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

##  Usage

### Running the Web App

```bash
streamlit run app.py
```

**Input the following details:**
- Location (e.g., Westlands, Kilimani, Karen)
- Property type (House, Apartment, Townhouse, Villa)
- Number of bedrooms and bathrooms
- Size in square meters
- Amenities (Security, Parking, Pool, Gym, etc.)

**Output:**
- Predicted price in KES
- Price range (± model error margin)
- Price breakdown and key drivers
- Explanation of factors affecting the price

### Training Models from Scratch

```bash
# Step 1: Scrape data (optional - data already provided)
python scrape.py

# Step 2: Clean data
python clean_data.py

# Step 3: EDA and baseline model
python day3_eda_baseline.py

# Step 4: Train advanced models
python day4_model_improvement.py
```

##  Project Structure

```
nairobi-house-price-prediction/
├── data/
│   ├── raw_listings.csv          # Scraped property data
│   ├── clean_listings.csv        # Cleaned dataset
│   ├── model.pkl                 # Trained XGBoost model
│   ├── feature_names.pkl         # Model features
│   └── model_comparison.csv      # Model performance metrics
├── visuals/
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── amenity_impact.png
│   └── location_impact.png
├── app.py                        # Streamlit web application
├── scrape.py                     # Web scraping script
├── clean_data.py                 # Data cleaning pipeline
├── day3_eda_baseline.py         # EDA and baseline model
├── day4_model_improvement.py    # Advanced models (RF, XGBoost)
├── requirements.txt              # Python dependencies
├── .python-version              # Python version for deployment
├── render.yaml                  # Render deployment config
└── README.md                    # This file
```

## Model Performance

### Best Model: XGBoost

| Metric | Value |
|--------|-------|
| **R² Score** | 0.383 |
| **MAE** | KES 29,992,354 |
| **RMSE** | KES 48,871,009 |
| **Variance Explained** | 38.3% |

### Model Comparison

| Model | Test R² | Test MAE (KES) |
|-------|---------|----------------|
| XGBoost | 0.383 | 29,992,354 |
| Linear Regression | 0.372 | 31,695,348 |
| Random Forest | 0.332 | 30,246,380 |

**Improvement over baseline:** 3% increase in R², 5.4% reduction in MAE

##  Deployment
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

### Deploy on Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Deploy from your repository

## Key Insights

### Top Price Drivers

1. **Size (Square Meters)** - Most important feature (24% importance)
2. **Location** - Creates up to 30x price difference between neighborhoods
3. **Amenities** - High-end amenities add significant value

### Location Impact

**Most Expensive Areas:**
- Muthaiga: KES 190M median
- Nyari: KES 157.5M median
- Thigiri: KES 140M median

**Most Affordable Areas:**
- Kitengela: KES 6.75M median
- Acacia: KES 6.2M median

**Price Multiplier:** Properties in Muthaiga cost 30.6x more than in Kitengela

### Amenity Impact

Properties with pools, gyms, and staff quarters (DSQ) command significantly higher prices. High amenity scores correlate with premium pricing.

##  Limitations

1. **Model Accuracy:** R² of 0.38 means the model explains only 38% of price variance. Predictions should be treated as rough estimates, not exact valuations.

2. **Average Error:** Predictions are off by ±KES 30M on average (about 49% of typical property prices).

3. **Missing Features:** The model doesn't account for:
   - Property condition and age
   - Exact micro-location quality
   - Recent renovations
   - Market timing and seasonality
   - Seller motivation

4. **Data Quality:** Limited to scraped data from one source (Property24.co.ke), which may not represent the entire market.

##  Future Improvements

- [ ] Collect more data (target 5,000+ properties)
- [ ] Add more features (property age, proximity to schools/hospitals, crime rates)
- [ ] Implement ensemble methods combining multiple models
- [ ] Add time-series analysis for market trends
- [ ] Create separate models for different property types
- [ ] Incorporate external data (economic indicators, infrastructure projects)
- [ ] Add confidence intervals to predictions
- [ ] Implement A/B testing for model improvements

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
6. 
##  Contact

**Moses Haggai Itapara**

- GitHub: [@YOUR_USERNAME](https://github.com/MosesItapara)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/Moses_Itapara)
- Email: haggaimoses19@gmail.com

##  Acknowledgments

- Data sourced from Property24.co.ke
- Built as part of a Data Science portfolio project
- Inspired by real-world property valuation challenges in Nairobi

---
