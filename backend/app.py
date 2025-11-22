from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import uvicorn
import io
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import shap
from textblob import TextBlob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = FastAPI()

# Add CORS middleware to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    # Read the uploaded file content
    contents = await file.read()
    
    # Determine file type and read accordingly
    file_extension = file.filename.split('.')[-1].lower() if file.filename else 'csv'
    
    try:
        if file_extension == 'xlsx' or file_extension == 'xls':
            # Read Excel file
            df = pd.read_excel(io.BytesIO(contents))
        else:
            # Read CSV file
            df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        return {"error": f"Failed to parse file: {str(e)}"}
    
    # Get headers
    headers = df.columns.tolist()
    
    # Convert dataframe to rows (list of lists)
    # Convert NaN values to None for JSON serialization
    rows = df.where(pd.notnull(df), None).values.tolist()
    
    # Detect column types
    detected_types = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        # Map pandas dtypes to frontend types
        if dtype.startswith('int') or dtype.startswith('float'):
            detected_types[col] = 'number'
        elif dtype == 'bool':
            detected_types[col] = 'boolean'
        elif dtype == 'object':
            # Check if it's boolean-like
            sample_values = df[col].dropna().head(10).astype(str).str.lower()
            if len(sample_values) > 0 and sample_values.isin(['yes', 'no', 'true', 'false', '1', '0']).all():
                detected_types[col] = 'boolean'
            else:
                detected_types[col] = 'text'
        else:
            detected_types[col] = 'text'
    
    return {
        'headers': headers,
        'rows': rows,
        'detectedTypes': detected_types
    }

class PredictRequest(BaseModel):
    csvData: Optional[Dict[str, Any]] = None
    fileName: Optional[str] = None
    columnMapping: Optional[Dict[str, str]] = None
    preprocessingOptions: Optional[Dict[str, bool]] = None

def reconstruct_dataframe(csv_data: Dict[str, Any]) -> pd.DataFrame:
    """Reconstruct pandas DataFrame from frontend CSV data structure."""
    if not csv_data or 'headers' not in csv_data or 'rows' not in csv_data:
        raise ValueError("Invalid CSV data structure")
    
    headers = csv_data['headers']
    rows = csv_data['rows']
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=headers)
    
    # Convert None back to NaN
    df = df.replace([None], np.nan)
    
    return df

def preprocess_data(df: pd.DataFrame, column_mapping: Dict[str, str], 
                   preprocessing_options: Dict[str, bool]) -> pd.DataFrame:
    """Preprocess the data according to user options."""
    df_processed = df.copy()
    
    # Handle missing values
    if preprocessing_options.get('handleMissing', True):
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        # Fill numeric columns with median
        if len(numeric_cols) > 0:
            imputer_numeric = SimpleImputer(strategy='median')
            df_processed[numeric_cols] = imputer_numeric.fit_transform(df_processed[numeric_cols])
        
        # Fill categorical columns with mode
        if len(categorical_cols) > 0:
            imputer_categorical = SimpleImputer(strategy='most_frequent')
            df_processed[categorical_cols] = imputer_categorical.fit_transform(df_processed[categorical_cols])
    
    # Encode categorical variables
    if preprocessing_options.get('encodeCategorical', True):
        label_encoders = {}
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != column_mapping.get('supportTextColumn', ''):
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le
    
    return df_processed

def generate_sentiment_score(text: str) -> float:
    """Generate sentiment score from text using TextBlob."""
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity  # Returns value between -1 (negative) and 1 (positive)
    except:
        return 0.0

def engineer_features(df: pd.DataFrame, column_mapping: Dict[str, str],
                     preprocessing_options: Dict[str, bool]) -> pd.DataFrame:
    """Create generic features: usage, milestone, sentiment, recency."""
    df_features = df.copy()
    
    # Feature 1: Usage metrics (from numeric columns)
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        # Create aggregated usage features
        df_features['feature_usage_mean'] = df_features[numeric_cols].mean(axis=1)
        df_features['feature_usage_sum'] = df_features[numeric_cols].sum(axis=1)
        df_features['feature_usage_max'] = df_features[numeric_cols].max(axis=1)
        df_features['feature_usage_min'] = df_features[numeric_cols].min(axis=1)
    
    # Feature 2: Milestone features (count of non-null values, indicating engagement)
    df_features['feature_milestone_count'] = df_features.notna().sum(axis=1)
    df_features['feature_milestone_ratio'] = df_features.notna().sum(axis=1) / len(df_features.columns)
    
    # Feature 3: Sentiment score (from support text column)
    if preprocessing_options.get('generateSentiment', False) and column_mapping.get('supportTextColumn'):
        text_col = column_mapping['supportTextColumn']
        if text_col in df_features.columns:
            df_features['feature_sentiment'] = df_features[text_col].apply(generate_sentiment_score)
        else:
            df_features['feature_sentiment'] = 0.0
    else:
        df_features['feature_sentiment'] = 0.0
    
    # Feature 4: Recency features (from date column)
    if column_mapping.get('dateColumn'):
        date_col = column_mapping['dateColumn']
        if date_col in df_features.columns:
            try:
                # Try to parse dates
                df_features[date_col] = pd.to_datetime(df_features[date_col], errors='coerce')
                current_date = pd.Timestamp.now()
                df_features['feature_recency_days'] = (current_date - df_features[date_col]).dt.days
                df_features['feature_recency_days'] = df_features['feature_recency_days'].fillna(0)
            except:
                df_features['feature_recency_days'] = 0
        else:
            df_features['feature_recency_days'] = 0
    else:
        df_features['feature_recency_days'] = 0
    
    return df_features

def prepare_target(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.Series:
    """Prepare target variable from churn column."""
    target_col = column_mapping.get('targetColumn', '')
    if not target_col or target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    target = df[target_col].copy()
    
    # Convert to binary (0/1)
    if target.dtype == 'object':
        target = target.astype(str).str.lower().str.strip()
        target = target.replace(['yes', 'true', '1', 'churn', 'y'], 1)
        target = target.replace(['no', 'false', '0', 'no churn', 'n', ''], 0)
        target = pd.to_numeric(target, errors='coerce').fillna(0)
    
    return target.astype(int)

def get_feature_importance_shap(model, X_train, X_test, feature_names):
    """Get SHAP values for feature importance."""
    try:
        # Use a subset for SHAP (it can be slow on large datasets)
        sample_size = min(100, len(X_test))
        X_sample = X_test[:sample_size]
        
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_sample)
        
        # Get mean absolute SHAP values per feature
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, get positive class
        
        mean_shap = np.abs(shap_values).mean(0)
        feature_importance = dict(zip(feature_names, mean_shap))
        
        return feature_importance
    except Exception as e:
        print(f"SHAP calculation error: {e}")
        # Fallback to model coefficients
        if hasattr(model, 'coef_'):
            coef_abs = np.abs(model.coef_[0])
            return dict(zip(feature_names, coef_abs))
        return {}

def generate_recommendations(prediction: int, probability: float, 
                           feature_importance: Dict[str, float],
                           column_mapping: Dict[str, str]) -> Dict[str, Any]:
    """Generate recommendations based on prediction and feature importance."""
    risk_level = 'high' if probability > 0.7 else 'medium' if probability > 0.5 else 'low'
    
    # Get top contributing features
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    reasons = [f"{feat.replace('feature_', '').replace('_', ' ').title()}: {imp:.2f}" 
               for feat, imp in top_features]
    
    # Generate actions based on risk and features
    actions = []
    if probability > 0.7:
        actions.append("Send personalized retention email")
        actions.append("Assign dedicated account manager")
        actions.append("Offer special discount or upgrade")
    elif probability > 0.5:
        actions.append("Send engagement survey")
        actions.append("Schedule check-in call")
    else:
        actions.append("Monitor usage patterns")
        actions.append("Send product tips newsletter")
    
    return {
        'prediction': 'Churn' if prediction == 1 else 'No Churn',
        'confidence': float(probability),
        'reasons': reasons,
        'action': ', '.join(actions),
        'risk': risk_level
    }

@app.post('/predict/')
async def predict(request: PredictRequest):
    """Main prediction endpoint implementing the full ML pipeline."""
    try:
        # Step 1: Reconstruct DataFrame
        if not request.csvData:
            return {"error": "No CSV data provided"}
        
        df = reconstruct_dataframe(request.csvData)
        column_mapping = request.columnMapping or {}
        preprocessing_options = request.preprocessingOptions or {}
        
        print(f"Processing {len(df)} rows with {len(df.columns)} columns")
        
        # Step 2: Prepare target variable
        try:
            y = prepare_target(df, column_mapping)
        except Exception as e:
            return {"error": f"Failed to prepare target variable: {str(e)}"}
        
        # Step 3: Preprocess data
        df_processed = preprocess_data(df, column_mapping, preprocessing_options)
        
        # Step 4: Engineer features
        df_features = engineer_features(df_processed, column_mapping, preprocessing_options)
        
        # Step 5: Prepare features for modeling (exclude target and ID columns)
        exclude_cols = [
            column_mapping.get('targetColumn', ''),
            column_mapping.get('customerIdColumn', ''),
            column_mapping.get('supportTextColumn', ''),
            column_mapping.get('dateColumn', '')
        ]
        exclude_cols = [col for col in exclude_cols if col in df_features.columns]
        
        X = df_features.drop(columns=exclude_cols, errors='ignore')
        
        # Ensure all columns are numeric
        X = X.select_dtypes(include=[np.number])
        
        if X.empty:
            return {"error": "No valid features found for prediction"}
        
        # Step 6: Scale features if requested
        scaler = None
        if preprocessing_options.get('applyScaling', True):
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        # Step 7: Train model
        # Split data for training
        if len(df) < 10:
            return {"error": "Insufficient data for training. Need at least 10 rows."}
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
        )
        
        # Train logistic regression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Step 8: Make predictions on all data
        probabilities = model.predict_proba(X)[:, 1]  # Probability of churn
        predictions = model.predict(X)
        
        # Step 9: Get SHAP explanations
        feature_importance = get_feature_importance_shap(
            model, X_train.values, X_test.values, X.columns.tolist()
        )
        
        # Step 10: Generate results for each customer
        results = []
        customer_id_col = column_mapping.get('customerIdColumn', '')
        
        for idx in range(len(df)):
            customer_id = df.iloc[idx][customer_id_col] if customer_id_col and customer_id_col in df.columns else f"Customer_{idx+1}"
            pred = int(predictions[idx])
            prob = float(probabilities[idx])
            
            recommendation = generate_recommendations(pred, prob, feature_importance, column_mapping)
            
            results.append({
                'customerId': str(customer_id),
                'churnProbability': prob,
                'prediction': recommendation['prediction'],
                'confidence': recommendation['confidence'],
                'risk': recommendation['risk'],
                'reasons': recommendation['reasons'],
                'recommendedActions': recommendation['action']
            })
        
        # Calculate overall statistics
        avg_churn_prob = float(probabilities.mean())
        churn_count = int(predictions.sum())
        
        return {
            'success': True,
            'summary': {
                'totalCustomers': len(df),
                'predictedChurn': churn_count,
                'averageChurnProbability': avg_churn_prob,
                'modelAccuracy': float(model.score(X_test, y_test)) if len(y_test) > 0 else 0.0
            },
            'predictions': results,
            'featureImportance': {k: float(v) for k, v in list(feature_importance.items())[:10]}
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Prediction error: {error_trace}")
        return {"error": f"Prediction failed: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)