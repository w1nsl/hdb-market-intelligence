import joblib
import pandas as pd
import os
from datetime import datetime

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