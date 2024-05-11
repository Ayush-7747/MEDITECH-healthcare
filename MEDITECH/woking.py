from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from pymongo import MongoClient

app = Flask(__name__, template_folder='templates')

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['patient_data']  # Replace 'your_database' with your actual database name
collection = db['patient_data']  # Replace 'your_collection' with your actual collection name

# Load the dataset
data = pd.read_csv(r'woking.py', delimiter=';')

# Data preprocessing
X = data.drop('cardio', axis=1)  # Features
y = data['cardio']  # Target variable

# Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('front.html')

# Define a route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    age = int(request.form['age'])*365
    gender = request.form['gender']
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    systolic_blood_pressure = float(request.form['systolic_blood_pressure'])
    diastolic_blood_pressure = float(request.form['diastolic_blood_pressure'])
    cholesterol = int(request.form['cholesterol'])
    glucose = int(request.form['glucose'])
    smoke = int(request.form['smoke'])
    alcohol=int(request.form['alcohol'])
    physical_activity = int(request.form['physical_activity'])
    cardiovascularhistory=int(request.form['cardio'])
    
    # Convert gender to numerical (assuming Male: 1, Female: 0)
    gender_num = 1 if gender == 'Male' else 0
    
    # Scale the input data
    new_patient_data = np.array([[age, gender_num, height, weight, systolic_blood_pressure, diastolic_blood_pressure, cholesterol, glucose, smoke,alcohol, physical_activity,cardiovascularhistory]])
    new_patient_data_scaled = scaler.transform(new_patient_data)  # Using the fitted scaler instance
    
    # Make prediction
    prediction = model.predict(new_patient_data_scaled)
    
    # Insert data into MongoDB
    data_entry = {
        'age': age,
        'gender': gender,
        'height': height,
        'weight': weight,
        'systolic_blood_pressure': systolic_blood_pressure,
        'diastolic_blood_pressure': diastolic_blood_pressure,
        'cholesterol': cholesterol,
        'glucose': glucose,
        'smoke': smoke,
        'alcohol': alcohol,
        'physical_activity': physical_activity,
        'cardiovascularhistory': cardiovascularhistory,
        'prediction': int(prediction[0])  # Convert prediction to int
    }
    collection.insert_one(data_entry)
    # Render the result template
    return render_template("result.html", prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
