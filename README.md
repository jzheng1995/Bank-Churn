# Predicting Bank Churn

## Project description

Welcome to the Customer Churn Prediction project! This repository contains the code, data, and documentation for a machine learning analysis aimed at predicting whether a customer will continue with their account or close it (i.e., churn). Accurately predicting customer churn is crucial for businesses to retain their customers and improve their overall performance.

## Objective

The main objective of this project is to develop a predictive model that can classify customers into two categories: those who will continue with their account and those who are likely to churn. By identifying at-risk customers, businesses can implement targeted interventions to retain them.

![Bank Churn](assets/bank.jpg)

## Dataset

The dataset used in this project belongs to the [Kaggle Playground Series](https://www.kaggle.com/competitions/playground-series-s4e1/overview). The data includes various features that might influence a customer's decision to stay or leave, such as:

Customer demographics (age, gender, location, etc.)
Account information (account balance, tenure, credit score, etc.)

## Methods

We evaluated the performance of the following models in predicting bank churn using Python and the scikit-learn machine learning framework:

- Logistic Classifier
- Random Forest Classifier
- XGBoost 
- Light GBM
- Voting Classifier

## Results

The final voting classifier model achieved 88% accuracy in both the public and private leaderboard submissions on Kaggle.
