# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 01:27:13 2024

@author: Nikhil Soni
student id: 22045846
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
dataset = "C:\data mining report\Sales_Transactions_Dataset_Weekly.csv"
df = pd.read_csv(dataset)

# Preprocessing
# Drop unnecessary columns or rows
# Handle missing values
# Encode categorical variables
# Split the data into training and test sets
X = df.drop(columns=['Product_Code', 'W51'])
y = df['W51']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train the ANN model
ann_model = MLPClassifier(hidden_layer_sizes=(100,),
                          max_iter=500, random_state=42)
ann_model.fit(X_train_scaled, y_train)
ann_predictions = ann_model.predict(X_test_scaled)

# Define and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Define and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# Evaluate the models
print("ANN Accuracy:", accuracy_score(y_test, ann_predictions))
print("ANN Classification Report:")
print(classification_report(y_test, ann_predictions))

print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_predictions))
print("Decision Tree Classification Report:")
print(classification_report(y_test, dt_predictions))
