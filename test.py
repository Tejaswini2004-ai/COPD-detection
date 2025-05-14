# import pandas as pd
# import joblib

# # Load the trained model and scaler
# model = joblib.load('copd_model.pkl')
# scaler = joblib.load('scaler.pkl')

# # Load new data for testing
# new_data = pd.read_csv(r'C:\Users\SRIVIDYA\Desktop\miniproject\data\finalalldata.csv')

# # 🔧 FIX: Drop non-numeric columns like uid, class, and label (if present)
# columns_to_drop = ['uid', 'class', 'label']
# new_data = new_data.drop(columns=[col for col in columns_to_drop if col in new_data.columns], errors='ignore')

# # 🔧 FIX: Keep only numeric columns
# new_data = new_data.select_dtypes(include=['number'])

# # Apply the same preprocessing steps (imputation, feature engineering, etc.)
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.decomposition import PCA

# # Preprocessing
# imputer = SimpleImputer(strategy='median')
# X_new = imputer.fit_transform(new_data)

# # Feature engineering
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X_poly = poly.fit_transform(X_new)

# pca = PCA(n_components=15)
# X_pca = pca.fit_transform(X_poly)

# # Scale new data
# X_scaled = scaler.transform(X_pca)

# # Predict
# predictions = model.predict(X_scaled)
# probabilities = model.predict_proba(X_scaled)[:, 1]

# # Output
# print("Predictions:", predictions)
# print("Probabilities of COPD:", probabilities)



import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import numpy as np

# Load the trained model, scaler, and PCA
model = joblib.load('copd_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

# Features (same as the features used for training the model)
features = ['sex', 'age', 'bmi', 'smoke', 'location', 'rs10007052', 'rs8192288', 
            'rs20541', 'rs12922394', 'rs2910164', 'rs161976', 'rs473892', 'rs159497', 'rs9296092']

# Preprocess input data
def preprocess_input(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data], columns=features)

    # Apply imputation
    imputer = SimpleImputer(strategy='median')
    X_new = imputer.fit_transform(input_df)

    # Apply Polynomial Feature transformation
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_new)

    # Apply PCA (transform with the saved PCA)
    X_pca = pca.transform(X_poly)

    # Scale the input data (using the saved scaler)
    X_scaled = scaler.transform(X_pca)
    
    return X_scaled

# Function to predict COPD risk
def predict_copd(user_input):
    # Preprocess the input data
    X_scaled = preprocess_input(user_input)
    
    # Predict using the trained model
    prediction = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)[:, 1]  # For class 1 (COPD)
    
    # Output the result
    if prediction[0] == 1:
        print("YES, you have a risk of COPD.")
    else:
        print("NO, you don't have a risk of COPD.")
    print(f"Prediction probability: {probability[0]:.4f}")

# Collect user inputs through terminal
def get_user_input():
    print("Enter the following details:")
    sex = int(input("Sex (0 for Female, 1 for Male): "))
    age = float(input("Age: "))
    bmi = float(input("BMI: "))
    smoke = int(input("Smoking Status (0 for Non-smoker, 1 for Smoker): "))
    location = float(input("Location: "))
    
    rs10007052 = float(input("rs10007052 genetic marker: "))
    rs8192288 = float(input("rs8192288 genetic marker: "))
    rs20541 = float(input("rs20541 genetic marker: "))
    rs12922394 = float(input("rs12922394 genetic marker: "))
    rs2910164 = float(input("rs2910164 genetic marker: "))
    rs161976 = float(input("rs161976 genetic marker: "))
    rs473892 = float(input("rs473892 genetic marker: "))
    rs159497 = float(input("rs159497 genetic marker: "))
    rs9296092 = float(input("rs9296092 genetic marker: "))
    
    # Store the user input in a list
    user_input = [sex, age, bmi, smoke, location, rs10007052, rs8192288, rs20541, 
                  rs12922394, rs2910164, rs161976, rs473892, rs159497, rs9296092]
    
    return user_input

# Main execution
def main():
    # Get user input from the terminal
    user_input = get_user_input()
    
    # Predict and display the result
    predict_copd(user_input)

# Run the program
if __name__ == "__main__":
    main()
