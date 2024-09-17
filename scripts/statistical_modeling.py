
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
import shap

def handle_missing_data(data):
    # Imputing missing values
    imputer = SimpleImputer(strategy='mean')
    data['TotalPremium'] = imputer.fit_transform(data[['TotalPremium']])
    data['TotalClaims'] = imputer.fit_transform(data[['TotalClaims']])
    return data

def encode_categorical_features(data):
    categorical_columns = ['IsVATRegistered', 'Citizenship', 'LegalType', 'Title', 'Language', 'Bank', 
                           'AccountType', 'MaritalStatus', 'Gender', 'Country', 'Province', 'MainCrestaZone', 
                           'SubCrestaZone', 'ItemType', 'VehicleType', 'Make', 'Model', 'Bodytype', 
                           'CoverCategory', 'CoverType', 'CoverGroup']
    
    for col in categorical_columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
    return data

def feature_engineering(data):
    # Feature engineering example
    data['ClaimsRatio'] = data['TotalClaims'] / data['TotalPremium']
    return data

def prepare_data(data):
    data = handle_missing_data(data)
    data = encode_categorical_features(data)
    data = feature_engineering(data)
    
    # Dropping unnecessary columns
    data = data.drop(columns=['UnderwrittenCoverID', 'PolicyID', 'TransactionMonth'], axis=1)
    
    return data



def train_test_split_data(data):
    X = data.drop(columns=['TotalPremium', 'TotalClaims'])
    y_premium = data['TotalPremium']
    y_claims = data['TotalClaims']
    
    X_train, X_test, y_premium_train, y_premium_test = train_test_split(X, y_premium, test_size=0.3, random_state=42)
    _, _, y_claims_train, y_claims_test = train_test_split(X, y_claims, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_premium_train, y_premium_test, y_claims_train, y_claims_test

def build_models(X_train, X_test, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'XGBoost': XGBRegressor()
    }
    
    results = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_train, model.predict(X_train))
        results[model_name] = mse
    
    return results



def evaluate_models(results):
    for model_name, mse in results.items():
        print(f"{model_name}: Mean Squared Error = {mse}")

def feature_importance_shap(model, X_train):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train)
