"""
Restaurant Rating Prediction - Python Script Version
====================================================

This script provides a streamlined version of the restaurant rating prediction model.
For full analysis and visualizations, please refer to the Jupyter notebook.

Author: Yaswanth Setty
Date: July 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path='Dataset .csv'):
    """Load and preprocess the restaurant data."""
    print("📊 Loading and preprocessing data...")
    
    # Load data
    data = pd.read_csv(file_path)
    print(f"Initial data shape: {data.shape}")
    
    # Drop unnecessary columns
    cols_to_drop = ['Restaurant ID', 'Rating color', 'Rating text', 'Address', 
                    'Restaurant Name', 'Locality Verbose']
    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])
    
    # Convert binary columns
    binary_columns = ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']
    for col in binary_columns:
        if col in data.columns:
            data[col] = data[col].map({'Yes': 1, 'No': 0})
    
    # Handle votes and create log transformation
    data['Votes'] = data['Votes'].fillna(0)
    data['log_votes'] = np.log1p(data['Votes'])
    
    # Handle cuisines
    data['Cuisines'] = data['Cuisines'].fillna('Missing')
    data['Cuisines_list'] = data['Cuisines'].apply(lambda x: [c.strip() for c in x.split(',')])
    
    # Multi-hot encode cuisines
    mlb = MultiLabelBinarizer()
    cuisine_encoded = mlb.fit_transform(data['Cuisines_list'])
    cuisine_df = pd.DataFrame(cuisine_encoded, columns=mlb.classes_, index=data.index)
    
    # Keep top 20 cuisines + Other
    cuisine_counts = cuisine_df.sum().sort_values(ascending=False)
    top_cuisines = cuisine_counts.head(20).index
    reduced_df = cuisine_df[top_cuisines].copy()
    reduced_df['Other'] = cuisine_df.drop(columns=top_cuisines).sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    
    # Merge back
    data = data.drop(columns=['Cuisines', 'Cuisines_list'])
    data = pd.concat([data, reduced_df], axis=1)
    
    # Group rare categories
    def group_rare_categories(series, min_freq=100):
        value_counts = series.value_counts()
        common = value_counts[value_counts >= min_freq].index
        return series.apply(lambda x: x if x in common else 'Other')
    
    if 'City' in data.columns:
        data['City'] = group_rare_categories(data['City'], min_freq=100)
    if 'Locality' in data.columns:
        data['Locality'] = group_rare_categories(data['Locality'], min_freq=100)
    
    # One-hot encode categorical variables
    categorical_cols = [col for col in ['Country Code', 'City', 'Locality', 'Currency'] if col in data.columns]
    if categorical_cols:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_cat = encoder.fit_transform(data[categorical_cols])
        encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_cols), index=data.index)
        data = data.drop(columns=categorical_cols)
        data = pd.concat([data, encoded_cat_df], axis=1)
    
    # Remove duplicates and filter zero ratings
    data = data.drop_duplicates()
    data = data[data['Aggregate rating'] > 0]
    
    print(f"Final data shape: {data.shape}")
    print("✅ Data preprocessing completed!")
    
    return data

def train_model(data):
    """Train the XGBoost model."""
    print("\n🤖 Training XGBoost model...")
    
    # Prepare features and target
    X = data.drop('Aggregate rating', axis=1)
    y = data['Aggregate rating']
    
    # Filter ratings that appear at least twice for stratification
    rating_counts = y.value_counts()
    valid_ratings = rating_counts[rating_counts >= 2].index
    filtered_data = data[data['Aggregate rating'].isin(valid_ratings)]
    
    X = filtered_data.drop('Aggregate rating', axis=1)
    y = filtered_data['Aggregate rating']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train XGBoost
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    xgb.fit(X_train, y_train)
    
    # Predictions
    y_pred = xgb.predict(X_test)
    
    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring='r2')
    print(f"\n🔄 Cross-Validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': xgb.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print(f"\n🎯 Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    return xgb, X_test, y_test, y_pred, feature_importance

def main():
    """Main execution function."""
    print("🍽️ Restaurant Rating Prediction Model")
    print("=" * 50)
    
    try:
        # Load and preprocess data
        data = load_and_preprocess_data()
        
        # Train model
        model, X_test, y_test, y_pred, feature_importance = train_model(data)
        
        print("\n✅ Model training completed successfully!")
        print("\n📝 To see detailed analysis and visualizations, run the Jupyter notebook:")
        print("   jupyter notebook 'Restaurant ratings.ipynb'")
        
        return model, feature_importance
        
    except FileNotFoundError:
        print("❌ Error: 'Dataset .csv' file not found!")
        print("Please ensure the dataset file is in the same directory.")
        return None, None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None, None

if __name__ == "__main__":
    model, feature_importance = main()
