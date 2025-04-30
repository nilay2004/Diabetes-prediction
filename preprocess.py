import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

def preprocess_data(input_file='data/diabetes.csv', output_dir='data/'):
    # Load dataset
    df = pd.read_csv(input_file)
    
    # Replace zeros with NaN for relevant columns
    columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)
    
    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    df[columns_with_zeros] = imputer.fit_transform(df[columns_with_zeros])
    
    # Features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save preprocessed data and scaler
    joblib.dump(scaler, f'{output_dir}scaler.pkl')
    joblib.dump(X_train_scaled, f'{output_dir}X_train_scaled.pkl')
    joblib.dump(X_test_scaled, f'{output_dir}X_test_scaled.pkl')
    joblib.dump(y_train, f'{output_dir}y_train.pkl')
    joblib.dump(y_test, f'{output_dir}y_test.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    preprocess_data()