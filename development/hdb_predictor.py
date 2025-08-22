import joblib
import pandas as pd
import os
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class HDBPricePredictor:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            # Find the predictor file's directory, then look for model directory
            predictor_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_dir = os.path.join(predictor_dir, "hdb_price_model")
        else:
            self.model_dir = model_dir
        self.model = None
        self.encoders = None
        self._load_model()
    
    def _load_model(self):
        try:
            model_path = os.path.join(self.model_dir, "model.pkl")
            encoders_path = os.path.join(self.model_dir, "label_encoders.pkl")
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            
            self.model = joblib.load(model_path)
            self.encoders = joblib.load(encoders_path)
            self.scaler = joblib.load(scaler_path)
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"🔍 Model directory: {os.path.abspath(self.model_dir)}")
            raise
    
    def predict_price(self, 
                     town: str,
                     flat_type: str,
                     floor_area_sqm: float,
                     storey_range: str,
                     flat_model: str,
                     remaining_lease: float,
                     lease_commence_date: int,
                     prediction_date: str = None):
        """
        Predict HDB resale price for given parameters.
        Args:
            town: Town name (e.g., 'TAMPINES', 'JURONG WEST')
            flat_type: Flat type (e.g., '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE')
            floor_area_sqm: Floor area in square meters
            storey_range: Storey range (e.g., '01 TO 03', '10 TO 12')
            flat_model: Flat model (e.g., 'Model A', 'New Generation', 'Improved')
            remaining_lease: Remaining lease in years (can be decimal)
            lease_commence_date: Year when lease commenced
            prediction_date: Date for prediction (YYYY-MM-DD format). If None, uses today's date.
        
        Returns:
            Predicted resale price
        """
        if self.model is None or self.encoders is None:
            raise ValueError("Model not loaded. Please initialize the predictor first.")
        
        try:
            if prediction_date is None:
                pred_date = datetime.now()
            else:
                pred_date = pd.to_datetime(prediction_date)
            
            storey_group = self._categorize_storey(storey_range)
            
            data = pd.DataFrame({
                'floor_area_sqm': [float(floor_area_sqm)],
                'remaining_lease': [float(remaining_lease)],
                'date_numeric': [(pred_date - pd.Timestamp('2015-01-01')).days],
                'building_age': [pred_date.year - int(lease_commence_date)],
                'storey_level': [float(storey_range.split(' TO ')[0])],
                'town_encoded': [self.encoders['town'].transform([town])[0]],
                'flat_type_encoded': [self.encoders['flat_type'].transform([flat_type])[0]],
                'flat_model_encoded': [self.encoders['flat_model'].transform([flat_model])[0]],
                'storey_group_encoded': [self.encoders['storey_group'].transform([storey_group])[0]]
            })
            
            prediction = self.model.predict(data)[0]
            return float(prediction)
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            raise
    
    def _categorize_storey(self, storey_range: str) -> str:
        """Categorize storey range into low, middle, or high."""
        if storey_range in ['01 TO 03', '04 TO 06', '07 TO 09']:
            return 'low'
        elif storey_range in ['10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21']:
            return 'middle'
        else:
            return 'high'


def predict_hdb_price(town: str,
                     flat_type: str,
                     floor_area_sqm: float,
                     storey_range: str,
                     flat_model: str,
                     remaining_lease: float,
                     lease_commence_date: int,
                     prediction_date: str = None,
                     model_dir: str = None):
    predictor = HDBPricePredictor(model_dir=model_dir)
    return predictor.predict_price(
        town=town,
        flat_type=flat_type,
        floor_area_sqm=floor_area_sqm,
        storey_range=storey_range,
        flat_model=flat_model,
        remaining_lease=remaining_lease,
        lease_commence_date=lease_commence_date,
        prediction_date=prediction_date
    )


def train_hdb_model():
    """
    Train the HDB price prediction model using the combined dataset.
    This function recreates the model training process from the Jupyter notebook.
    """
    print("🚀 Starting HDB price prediction model training...")
    
    # Load and combine datasets
    print("📊 Loading HDB datasets...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    df15to16_path = os.path.join(project_dir, 'Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv')
    df17onwards_path = os.path.join(project_dir, 'Resale flat prices based on registration date from Jan-2017 onwards.csv')
    
    df15to16 = pd.read_csv(df15to16_path)
    df17onwards = pd.read_csv(df17onwards_path)
    
    # Convert remaining lease to float for 2017+ data
    def convert_lease_to_float(lease_str):
        if pd.isna(lease_str):
            return None
        years = 0
        months = 0
        if 'years' in lease_str:
            years_part = lease_str.split('years')[0].strip()
            if 'year' in years_part:
                years_part = years_part.split('year')[0].strip()
            years = int(years_part)
        
        if 'months' in lease_str:
            months_part = lease_str.split('years')[-1].split('months')[0].strip()
            if 'month' in months_part:
                months_part = months_part.split('month')[0].strip()
            if months_part:
                months = int(months_part)
        
        return years + (months / 12)
    
    df17onwards['remaining_lease'] = df17onwards['remaining_lease'].apply(convert_lease_to_float)
    
    # Combine datasets
    df_combined = pd.concat([df15to16, df17onwards], ignore_index=True)
    print(f"✅ Combined dataset loaded: {len(df_combined):,} records")
    
    # Feature engineering
    print("🔧 Engineering features...")
    df_combined['month_date'] = pd.to_datetime(df_combined['month'])
    df_combined['date_numeric'] = (df_combined['month_date'] - pd.Timestamp('2015-01-01')).dt.days
    df_combined['building_age'] = df_combined['month_date'].dt.year - df_combined['lease_commence_date']
    df_combined['storey_level'] = df_combined['storey_range'].str.split(' TO ').str[0].astype(int)
    
    # Categorize storey
    def categorize_storey(storey_range):
        if storey_range in ['01 TO 03', '04 TO 06', '07 TO 09']:
            return 'low'
        elif storey_range in ['10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21']:
            return 'middle'
        else:
            return 'high'
    
    df_combined['storey_group'] = df_combined['storey_range'].apply(categorize_storey)
    
    # Prepare features
    numerical_features = ['floor_area_sqm', 'remaining_lease', 'date_numeric', 'building_age', 'storey_level']
    categorical_features = ['town', 'flat_type', 'flat_model', 'storey_group']
    
    # Encode categorical variables
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        df_combined[f'{feature}_encoded'] = le.fit_transform(df_combined[feature])
        label_encoders[feature] = le
    
    feature_columns = numerical_features + [f'{feature}_encoded' for feature in categorical_features]
    X = df_combined[feature_columns]
    y = df_combined['resale_price']
    
    print(f"✅ Features prepared: {X.shape[1]} features, {len(X):,} samples")
    
    # Split data
    print("📊 Splitting data (80% train, 10% val, 10% test)...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples") 
    print(f"Test set: {len(X_test):,} samples")
    
    # Train Random Forest model
    print("🌲 Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("📊 Evaluating model performance...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ Model Performance:")
    print(f"   RMSE: ${rmse:,.0f}")
    print(f"   MAE: ${mae:,.0f}")
    print(f"   R²: {r2:.4f}")
    
    # Save model
    print("💾 Saving model...")
    model_dir = os.path.join(script_dir, "hdb_price_model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model components
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(label_encoders, os.path.join(model_dir, "label_encoders.pkl"))
    
    # Create dummy scaler for compatibility (Random Forest doesn't need scaling)
    from sklearn.preprocessing import StandardScaler
    dummy_scaler = StandardScaler()
    dummy_scaler.fit(X_train[numerical_features])  # Fit on numerical features only
    joblib.dump(dummy_scaler, os.path.join(model_dir, "scaler.pkl"))
    
    # Save model metadata
    model_info = {
        "model_type": "Random Forest",
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "performance": {
            "test_rmse": float(rmse),
            "test_mae": float(mae),
            "test_r2": float(r2)
        },
        "training_data_shape": list(X.shape),
        "date_range": {
            "start": df_combined['month_date'].min().strftime('%Y-%m-%d'),
            "end": df_combined['month_date'].max().strftime('%Y-%m-%d')
        }
    }
    
    with open(os.path.join(model_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✅ Model saved to: {model_dir}")
    print("🎉 Model training completed successfully!")
    
    return model, label_encoders, model_info


if __name__ == "__main__":
    # Train model when script is run directly
    train_hdb_model()