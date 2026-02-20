import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page Config
st.set_page_config(
    page_title="Nairobi House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Loading model and feature names
@st.cache_resource
def load_model():
    with open('data/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('data/featurre_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names

model, feature_names = load_model()

## Loading data for reference to get location options
@st.cache_data
def load_reference_data():
    df = pd.read_csv('data/clean_listings.csv')
    sale_df = df[df['Listing_Type'] == 'For Sale']
    return sale_df

ref_df = load_reference_data()

top_locations = ref_df['Location'].value_counts().head(15).index.tolist()
all_locations = sorted(top_locations + ['Other'])


## Title and Description
st.title("🏠 Nairobi House Price Predictor")
st.markdown("""
Predict property prices in Nairobi based on location, size and amenities.
This tool uses machine learning trained on real market data.
""")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Property Details")

    location = st.selectbox(
        "Location",
        options = all_locations,
        help = "Select the neighbourhood or area"
    )

    property_type = st.selectbox(
        "Property Type",
        options = ['House', 'Apartment', 'Townhouse', 'Villa'],
        help = "Type of property"
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of bedroooms"
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        help="Number of bathrooms"
    )

with col2:
    st.subheader("Size & Features")

    size_sqm = st.number_input(
        "Size (Square Metres)",
        min_value=20,
        max_value=2000,
        value=120,
        help="Total floor area in square metres"
    )

    st.markdown("**Select Amenities:**")
    has_security = st.checkbox("Security", value=True)
    has_parking = st.checkbox("Parking", value=True)
    has_garden = st.checkbox("Garden", value=False)
    has_pool = st.checkbox("Swimming Pool", value=False)
    has_gym = st.checkbox("Gym", value=False)
    has_generator = st.checkbox("Generator/Backup Power", value=False)
    has_borehole = st.checkbox("Borehole", value=False)
    has_dsq = st.checkbox("DSQ (Staff Quarters)", value=False)

amenity_score = 0
if has_security: amenity_score += 1
if has_parking: amenity_score += 1
if has_garden: amenity_score += 2
if has_pool: amenity_score += 3
if has_gym: amenity_score += 3
if has_generator: amenity_score += 3
if has_borehole: amenity_score += 3
if has_dsq: amenity_score += 3

distance_map = {
    'Westlands': 5.2, 'Kilimani': 3.8, 'Karen': 13.5, 'Lavington': 4.5,
    'Kileleshwa': 4.2, 'Runda': 9.8, 'Riverside': 2.5, 'Parklands': 4.8,
    'Muthaiga': 7.2, 'Langata': 11.3, 'South C': 6.5, 'South B': 5.8,
    'Upperhill': 2.1, 'Hurlingham': 4.0, 'Gigiri': 8.5, 'Kitisuru': 10.2,
    'Other': 10.0
}
distance_to_cbd = distance_map.get(location, 10.0)

# Current month
listing_month = pd.Timestamp.now().month

st.markdown("---")

if st.button("🔮 Predict Price", type="primary", use_container_width=True):

    input_data = {
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Size_SQM': size_sqm,
        'Amenity_Score': amenity_score,
        'Distance_to_CBD_KM': distance_to_cbd,
        'Listing_Month': listing_month
    }

    for loc in top_locations:
        col_name = f'Loc_{loc}'
        input_data[col_name] = 1 if location == loc else 0
    input_data['Loc_Other'] = 1 if location == 'Other'else 0


    for ptype in ['Apartment', 'House', 'Townhouse', 'Villa']:
        col_name = f'Type_{ptype}'
        input_data[col_name] = 1 if property_type == ptype else 0

    input_df = pd.DataFrame([input_data])

    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0

    input_df = input_df[feature_names]

    prediction = model.predict(input_df)[0]

    mae = 29_992_354
    lower_bound = max(0, prediction - mae)
    upper_bound = prediction + mae

    st.markdown("---")
    st.subheader("💰 Predicted Price")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.metric(
            label="Estimated Price",
            value=f"KES {prediction:,.0f}",
            help="Predicted market price based on your inputs"
        )

        st.caption(f"**Range:** KES {lower_bound:,.0f} - KES {upper_bound:,.0f}")
        st.caption(f"*In Millions:* KES {prediction/1_000_000:.2f}M")

    st.markdown("---")
    st.subheader("📊 Price Breakdown")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Key Factors:**")
        st.write(f"• Location: {location}")
        st.write(f"• Size: {size_sqm} m²")
        st.write(f"• Price/m²: KES {prediction/size_sqm:,.0f}")
    
    with col2:
        st.markdown("**Property Features:**")
        st.write(f"• {bedrooms} Bedroom(s)")
        st.write(f"• {bathrooms} Bathroom(s)")
        st.write(f"• Amenity Score: {amenity_score}/20")
    
    with col3:
        st.markdown("**Location:**")
        st.write(f"• {location}")
        st.write(f"• Distance to CBD: {distance_to_cbd} km")
        st.write(f"• Type: {property_type}")
    
    # What affects the price
    st.markdown("---")
    st.subheader(" What's Driving This Price?")

    st.markdown("""
    **Based on our model, the main price drivers are:**
    
    1. **Size (Square Meters)** - Larger properties command higher prices
    2. **Location** - Premium neighborhoods like Muthaiga and Runda are 30x more expensive than Kitengela
    3. **Amenities** - Pools, gyms, and DSQs add significant value
    """)
    
    # Confidence indicator
    st.markdown("---")
    st.info("""
    ** About This Prediction**
    
    This model explains 38% of price variance with an average error of ±KES 30M.
    Actual prices may vary based on:
    - Property condition and age
    - Exact location within the neighborhood  
    - Market conditions at time of sale
    - Negotiation and seller motivation
    
    Use this as a rough estimate, not a guaranteed valuation.
    """)

st.sidebar.header("Model Information")
st.sidebar.markdown("""
**Model:** XGBoost
**Accuracy:** R^2 = 0.38
**Avg Error:** +/- KES 30M
**Training Data:** 1,840 properties
""")

st.sidebar.markdown("---")
st.sidebar.header(" Location Price Guide")

# Show median prices by location
location_prices = ref_df.groupby('Location')['Price_KES'].median().sort_values(ascending=False)
st.sidebar.markdown("**Most Expensive:**")
for loc, price in location_prices.head(5).items():
    st.sidebar.write(f"• {loc}: KES {price/1_000_000:.1f}M")

st.sidebar.markdown("**Most Affordable:**")
for loc, price in location_prices.tail(5).items():
    st.sidebar.write(f"• {loc}: KES {price/1_000_000:.1f}M")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**💻 Project Info**  
Built as part of a Data Science portfolio project.  
Data: Scraped from Kenyan real estate sites  
[View on GitHub](https://github.com/YOUR_USERNAME/NAIROBIHOUSEPREDICTION)
""")
