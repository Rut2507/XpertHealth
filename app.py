import csv
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import diseaseprediction

app = Flask(__name__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix

# for DIABETES PREDICTION
data = pd.read_csv('diabetes_data.csv')
data.drop(['SkinThickness', 'DiabetesPedigreeFunction'], axis=1, inplace=True)
X = data.drop('Outcome', axis=1)
y = data['Outcome']
# splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# SMOTE applied for class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# training a Random_Forest model
model = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100,random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# prediction function
def diabetes_diagnosis(patient_data):
    prediction_probabilities = model.predict_proba(patient_data)
    #  Prob of class 1 = high risk of diabetes
    prediction_probability = prediction_probabilities[0][1]

    if prediction_probability >= 0.5:
        result = f"High diabetes risk.{prediction_probability:.1%} Please consult a healthcare professional."
    else:
        result = f"No diabetes risk.   Prediction Probability: {prediction_probability:.1%}"

    return result


@app.route('/diabetesPredict', methods=['GET', 'POST'])
def diabetesPredict():
    if request.method == 'POST':
        # Get input data and create df
        input_data = {
            'Pregnancies': [int(request.form['pregnancies'])],
            'Glucose': [int(request.form['glucose'])],
            'BloodPressure': [int(request.form['blood_pressure'])],
            'Insulin': [int(request.form['insulin'])],
            'BMI': [float(request.form['bmi'])],
            'Age': [int(request.form['age'])]
        }
        patient_data = pd.DataFrame(input_data)

        # prediction based on input using the prediction function
        result = diabetes_diagnosis(patient_data)
        return render_template('Diabetes_result.html', result=result)

    return render_template('includes/Diabetes_predict.html')

with open('templates/Testing.csv', newline='') as f:
    reader = csv.reader(f)
    symptoms = next(reader)
    symptoms = symptoms[:len(symptoms) - 1]


@app.route('/symptomchecker', methods=['GET'])
def dropdown():
    return render_template('includes/sympCheck.html', symptoms=symptoms)


@app.route('/disease_predict', methods=['POST'])
def disease_predict():
    selected_symptoms = []
    if (request.form['Symptom'] != "") and (request.form['Symptom'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom'])
    if (request.form['Symptom2'] != "") and (request.form['Symptom2'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom2'])
    if (request.form['Symptom3'] != "") and (request.form['Symptom3'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom3'])
    if (request.form['Symptom4'] != "") and (request.form['Symptom4'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom4'])
    if (request.form['Symptom5'] != "") and (request.form['Symptom5'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom5'])

    disease = diseaseprediction.dosomething(selected_symptoms)
    return render_template('disease_predict.html', disease=disease, symptoms=symptoms)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/services')
def services():
    return render_template('index.html')

@app.route('/diseases')
def diabetes():
    return render_template('diseases.html')

@app.route('/acidity')
def acidity():
    return render_template('acidity.html')

@app.route('/acne')
def acne():
    return render_template('acne.html')

@app.route('/allergy')
def allergy():
    return render_template('allergy.html')

@app.route('/anxiety')
def anxiety():
    return render_template('anxiety.html')

@app.route('/bodyache')
def bodyache():
    return render_template('bodyache.html')

@app.route('/catract')
def catract():
    return render_template('catract.html')

@app.route('/conjuctivitis')
def conjuctivitis():
    return render_template('conjuctivitis.html')

@app.route('/depresion')
def depresion():
    return render_template('depresion.html')

@app.route('/fever')
def fever():
    return render_template('fever.html')

@app.route('/foodpoision')
def foodpoision():
    return render_template('foodpoision.html')

@app.route('/kidneystone')
def kidneystone():
    return render_template('kidneystone.html')

@app.route('/lungcancer')
def lungcancer():
    return render_template('lungcancer.html')

@app.route('/nausea')
def nausea():
    return render_template('nausea.html')

if __name__ == '__main__':
    app.run(debug=True)
