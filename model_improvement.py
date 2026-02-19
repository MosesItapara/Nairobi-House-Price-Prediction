import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/clean_listings.csv')
print(f"Loaded {len(df)} listings")

sale_df = df[df['Listing_Type'] == 'For Sale'].copy()
print(f"\nFocusing on For Sale properties: {len(sale_df)} listings")

numeric_features = ['Bedrooms', 'Bathrooms', 'Size_SQM', 'Amenity_Score', 
                   'Distance_to_CBD_KM', 'Listing_Month']

top_locations = sale_df['Location'].value_counts().head(15).index
sale_df['Location_Group'] = sale_df['Location'].apply(
    lambda x: x if x in top_locations else 'Other'
)

location_dummies = pd.get_dummies(sale_df['Location_Group'], prefix='Loc')

property_dummies = pd.get_dummies(sale_df['Property_Type'], prefix='Type')

X = pd.concat([
    sale_df[numeric_features],
    location_dummies,
    property_dummies
], axis = 1)

y = sale_df['Price_KES']

print(f"\nTotal features: {X.shape[1]}")
print(f"Training samples: {len(X)}")

X_train, X_test, y_train, y_test =  train_test_split( X, y, test_size=0.2, random_state=42)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

### RANDOM FOREST

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("Training complete")

rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
rf_test_mae = mean_absolute_error(y_test, rf_test_pred)
rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
rf_train_r2 = r2_score(y_train, rf_train_pred)
rf_test_r2 = r2_score(y_test, rf_test_pred)

print(f"\nMAE")
print(f" Train: KES {rf_train_mae:,.0f}")
print(f" Test: KES {rf_test_mae:,.0f}")
print(f"\n Predictions are off by KES {rf_test_mae:,.0f} on average")
print(f"  That's {(rf_test_mae / y_test.mean()) * 100:.1f}% of average price")

print(f"\nRMSE:")
print(f"  Train: KES {rf_train_rmse:,.0f}")
print(f"  Test:  KES {rf_test_rmse:,.0f}")

print(f"\nR² Score:")
print(f"  Train: {rf_train_r2:.3f}")
print(f"  Test:  {rf_test_r2:.3f}")
print(f"  Model explains {rf_test_r2*100:.1f}% of price variance")


### XGBOOST

xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)
print("Training complete")

xgb_train_pred = xgb_model.predict(X_train)
xgb_test_pred = xgb_model.predict(X_test)

xgb_train_mae = mean_absolute_error(y_train, xgb_train_pred)
xgb_test_mae = mean_absolute_error(y_test, xgb_test_pred)
xgb_train_rmse = np.sqrt(mean_squared_error(y_train, xgb_train_pred))
xgb_test_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_pred))
xgb_train_r2 = r2_score(y_train, xgb_train_pred)
xgb_test_r2 = r2_score(y_test, xgb_test_pred)

print(f"\nMAE:")
print(f"  Train: KES {xgb_train_mae:,.0f}")
print(f"  Test:  KES {xgb_test_mae:,.0f}")
print(f"\n  Predictions are off by KES {xgb_test_mae:,.0f} on average")
print(f"That's {(xgb_test_mae / y_test.mean()) * 100:.1f}% of average price")

print(f"\nRMSE:")
print(f"  Train: KES {xgb_train_rmse:,.0f}")
print(f"  Test:  KES {xgb_test_rmse:,.0f}")

print(f"\nR² Score:")
print(f"  Train: {xgb_train_r2:.3f}")
print(f"  Test:  {xgb_test_r2:.3f}")
print(f"  Model explains {xgb_test_r2*100:.1f}% of price variance")

### COMPARISON

baseline_df = pd.read_csv('data/model_results.csv')

comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'Test_MAE': [baseline_df['Test_MAE'].iloc[0], rf_test_mae, xgb_test_mae],
    'Test_RMSE': [baseline_df['Test_RMSE'].iloc[0], rf_test_rmse, xgb_test_rmse],
    'Test_R2': [baseline_df['Test_R2'].iloc[0], rf_test_r2, xgb_test_r2],
})

comparison = comparison.sort_values('Test_R2', ascending=False)

print("\n")
print(comparison.to_string(index=False))

best_model_name = comparison.iloc[0]['Model']
best_r2 = comparison.iloc[0]['Test_R2']
best_mae = comparison.iloc[0]['Test_MAE']

print(f"BEST MODEL: {best_model_name}")
print(f"R2 Score: {best_r2:.3f} ({best_r2*100:.1f}% varaince explained)")
print(f"MAE: KES {best_mae:,.0f}")

baseline_r2 = baseline_df['Test_R2'].iloc[0]
baseline_mae = baseline_df['Test_MAE'].iloc[0]
r2_improvement = ((best_r2 - baseline_r2) / baseline_r2) * 100
mae_improvement = ((baseline_mae - best_mae) / baseline_mae) * 100

print(f"\nImprovements over baseline:")
print(f"  R² improved by {r2_improvement:.1f}%")
print(f"  MAE reduced by {mae_improvement:.1f}%")

comparison.to_csv('data/model_comparison.csv', index=False)
print(f"\nSaved: data/model_comparison.csv")

### TOP 5 PRICE DRIVERS

if best_model_name == 'Random Forest':
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': xgb_model.feature_importances_
    })

else:
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': xgb_model.feature_importances_
    })

feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nTop 5 Most Important Features:")
for i, row in enumerate(feature_importance.head(5).iterrows(), 1):
    idx, data = row
    print(f"  {i}. {data['Feature']:30s} - {data['Importance']:.4f}")

fig, ax = plt.subplots(figsize=(10, 8))
top_15 = feature_importance.head(15)
ax.barh(range(len(top_15)), top_15['Importance'])
ax.set_yticks(range(len(top_15)))
ax.set_yticklabels(top_15['Feature'])
ax.set_xlabel('Importance Score')
ax.set_title(f'Top 15 Features by Importance ({best_model_name})')
ax.invert_yaxis()
plt.tight_layout()

#### AMENITY IMPACT

amenity_impact = sale_df.groupby('Amenity_Score').agg({
    'Price_KES': ['mean','median', 'count']
}).round(0)
amenity_impact.columns = ['Mean_Price', 'Median_Price', 'Count']
amenity_impact  =amenity_impact[amenity_impact['Count'] >= 10]

print("\nMedian Price by Amenity Score:")
for score, row in amenity_impact.iterrows():
    print(f"  Score {score}: KES {row['Median_Price']:>15,.0f} ({int(row['Count'])} properties)")

# Calculate percentage increase
if len(amenity_impact) >= 2:
    baseline_price = amenity_impact.iloc[0]['Median_Price']
    high_amenity_price = amenity_impact.iloc[-1]['Median_Price']
    increase = ((high_amenity_price - baseline_price) / baseline_price) * 100
    print(f"\n Properties with high amenity scores cost {increase:.1f}% more")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(amenity_impact.index, amenity_impact['Median_Price'] / 1_000_000,
        marker='o', linewidth=2, markersize=10, color='green')
ax.set_xlabel('Amenity Score')
ax.set_ylabel('Median Price (Millions KES)')
ax.set_title('Impact of Amenities on Property Price')
ax.grid(alpha=0.3)
plt.tight_layout()

#### LOCATION IMPACT 

location_impact = sale_df.groupby('Location').agg({
    'Price_KES': ['median', 'count']
})
location_impact.columns = ['Median_Price', 'Count']
location_impact = location_impact[location_impact['Count'] >= 5]
location_impact = location_impact.sort_values('Median_Price', ascending=False)

print("\nTop 5 Most Expensive Locations:")
for i, (loc, row) in enumerate(location_impact.head(5).iterrows(), 1):
    print(f"  {i}. {loc:20s} - KES {row['Median_Price']:>15,.0f} ({int(row['Count'])} properties)")

print("\nMost Affordable Locations:")
for i, (loc, row) in enumerate(location_impact.tail(3).iterrows(), 1):
    print(f"  {i}. {loc:20s} - KES {row['Median_Price']:>15,.0f} ({int(row['Count'])} properties)")

# Price difference
most_expensive = location_impact.iloc[0]['Median_Price']
least_expensive = location_impact.iloc[-1]['Median_Price']
multiplier = most_expensive / least_expensive
print(f"\n Most expensive location costs {multiplier:.1f}x more than least expensive")

fig, ax = plt.subplots(figsize=(12, 6))
top_10_loc = location_impact.head(10)
ax.barh(range(len(top_10_loc)), top_10_loc['Median_Price'] / 1_000_000)
ax.set_yticks(range(len(top_10_loc)))
ax.set_yticklabels(top_10_loc.index)
ax.set_xlabel('Median Price (Millions KES)')
ax.set_title('Top 10 Most Expensive Locations')
ax.invert_yaxis()
plt.tight_layout()

### SAVING BEST MODEL

if best_model_name == 'Random Forest':
    best_model = rf_model
else:
    best_model = xgb_model

with open('data/model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('data/featurre_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print(f"\n Saved best model: {best_model_name}")
print("  Files:")
print("    - data/model.pkl")
print("    - data/feature_names.pkl")

#### SUMMARY

print(f"\nBest Model: {best_model_name}")
print(f"   Test R²: {best_r2:.3f} ({best_r2*100:.1f}% variance explained)")
print(f"   Test MAE: KES {best_mae:,.0f}")

print(f"\nImprovements over Baseline:")
print(f"R² improved: +{r2_improvement:.1f}%")
print(f"MAE reduced: -{mae_improvement:.1f}%")

print("\nKey Findings:")
print(f"1. Top price driver: {feature_importance.iloc[0]['Feature']}")
if len(amenity_impact) >= 2:
    print(f"2. High amenities add +{increase:.1f}% to price")
print(f"3. Location creates {multiplier:.1f}x price difference")