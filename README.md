# Lung_Cancer_Detection
 Dataset : https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset
:

🫁 Lung Cancer Detection System

A Machine Learning–based Web Application using Flask

This project is a Lung Cancer Detection System that uses a trained Machine Learning model to predict whether a patient may have lung cancer based on input medical features. The system includes both an ML backend and a clean, user-friendly Flask web interface for users to upload or enter data and receive predictions instantly.

🧠 Project Overview

This project combines machine learning, data preprocessing, and a Flask web framework to create an end-to-end lung cancer prediction application.
The system analyzes patient data using a trained ML model and classifies whether the person is at High Risk, Medium Risk, or Low Risk of lung cancer.

🚀 Key Features

✔️ ML Model Training (Logistic Regression / Random Forest / SVM – choose your algorithm)

✔️ Data Cleaning & Preprocessing

✔️ Feature Engineering for better performance

✔️ Model Evaluation using accuracy, confusion matrix, precision & recall

✔️ Flask-based Web Interface

✔️ User can input symptoms or medical parameters

✔️ System displays prediction results with risk level

✔️ Lightweight, fast, and easy to deploy

🛠️ Technologies Used
Machine Learning

Python

NumPy

Pandas

Scikit-Learn

Matplotlib / Seaborn (for analysis)

Web Application

Flask

HTML / CSS

Bootstrap

Jinja2 Templates

📁 Project Structure

🔍 How It Works

The dataset is loaded and cleaned (handling missing values & encoding).

Model is trained using machine learning algorithms.

The best model is saved using pickle (.pkl).

Flask loads the trained model during runtime.

User enters medical details in the web interface.

Flask sends the input to the ML model.

The model predicts cancer probability and returns results to the UI.

🧪 Model Training Summary

Trained multiple classifiers

Compared them using performance metrics

Selected the model with the highest accuracy

Saved final trained model for deployment

🌐 Web Interface

The application provides:

Simple input form for users

Attractive UI designed with HTML, CSS, and Bootstrap

Output page showing prediction & risk level

Completely browser-based workflow

📦 Installation

Run the following commands:

pip install -r requirements.txt
python app.py


Open in browser:

http://127.0.0.1:5000

📊 Results

The ML model provides:

Cancer detection classification

Probability score

User-friendly interpretation (Low / Medium / High Risk)

📌 Conclusion

This project demonstrates how machine learning can be integrated with Flask to create a practical medical prediction system. It can be extended further with real hospital-grade datasets, image-based CT scan analysis, and cloud deployment.
