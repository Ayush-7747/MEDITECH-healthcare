
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (replace 'data.csv' with your dataset)
data = pd.read_csv(r'cardio_train.csv', delimiter=';')
print(data.head())


# Data preprocessing
X = data.drop('cardio', axis=1) # Features
y = data['cardio'] # Target variable

# Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Example usage of the trained model for prediction
new_patient_data = np.array([[20228,1,156,85.0,140,90,3,1,0,0,1,1]]) # New patient's clinical characteristics
new_patient_data_scaled = scaler.transform(new_patient_data)
prediction = model.predict(new_patient_data_scaled)
print("Predicted class for the new patient:", prediction)
