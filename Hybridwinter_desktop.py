import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import load_model
import sys


def load_models_and_scaler():
    best_rf = joblib.load('random_forest_model_WinterMonths2dayFortLee_updated.pkl')
    best_xgb = joblib.load('xgboost_model_WinterMonths2dayFortLee.pkl')
    best_ann = load_model('ann_model_WinterMonths2dayFortLee.keras')
    scaler = joblib.load('scaler_rf.pkl')  
    return best_rf, best_xgb, best_ann, scaler

def predict(user_input):
    best_rf, best_xgb, best_ann, scaler = load_models_and_scaler()
    
    # Scale user input
    user_input_scaled = scaler.transform([user_input])
    
    # Make predictions
    rf_pred = best_rf.predict(user_input_scaled)[0]
    xgb_pred = best_xgb.predict(user_input_scaled)[0]
    ann_pred = best_ann.predict(user_input_scaled)[0][0]
    
    return rf_pred, xgb_pred, ann_pred
    
if __name__ == "__main__":
    '''
    if len(sys.argv) != 10:  # 9 features + 1 for the script name
        print("Error: Expected 9 input features.")
        sys.exit(1)
    '''
    # Convert command-line arguments to float and create the user_input list
    #user_input = [float(arg) for arg in sys.argv[1:]]
    # Define features

    user_input = [13, 7.2, 16.3, 178.6, 14.8, 0, 103, 58.2, 67.6]
    features = ['pm25-1', 'temp', 'visibility', 'winddir', 'windspeed', 'precip', 'solarradiation', 'cloudcover', 'humidity']

    rf_pred, xgb_pred, ann_pred = predict(user_input)
    accuracy_metrics_rf = joblib.load('model_accuracy_metrics_rf.pkl')
    accuracy_metrics_xg = joblib.load('model_accuracy_metrics_xg.pkl')
    accuracy_metrics_ann = joblib.load('model_accuracy_metrics_ann.pkl')

    # Display user input predictions and confidence
    print(f"Random Forest User Prediction: {rf_pred}, accuracy: {accuracy_metrics_rf:.2f}")
    print(f"XGBoost User Prediction: {xgb_pred}, accuracy: {accuracy_metrics_xg:.2f}")
    print(f"ANN User Prediction: {ann_pred}, accuracy: {accuracy_metrics_ann:.2f}")


    accuracies = {accuracy_metrics_rf:(rf_pred, "RF"), accuracy_metrics_xg:(xgb_pred, "XG"), accuracy_metrics_ann:(ann_pred, "ANN")}
    keymax = max(accuracies)

    #print(accuracies[keymax][1])
    print(f"The best model is: {accuracies[keymax][1]}, Forecasted = {accuracies[keymax][0]:.2f}, Accuracy = {keymax:.2f}")


