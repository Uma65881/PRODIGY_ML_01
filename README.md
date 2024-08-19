# PRODIGY_ML_01
House Price Prediction using Linear Regression
This repository contains the implementation of a linear regression model to predict house prices based on their living area, number of bedrooms, and number of bathrooms. The dataset used for this project is from the Kaggle competition "House Prices - Advanced Regression Techniques."

Table of Contents
Overview
Dataset
Project Structure
Installation
Usage
Model Evaluation
Results
Visualization
Contributing
License
Overview
The goal of this project is to develop a linear regression model to estimate the price of houses based on specific features. The model is trained using a subset of features from the dataset and evaluated on a test set.

Dataset
The dataset is sourced from the Kaggle House Prices - Advanced Regression Techniques competition.

Features Used:
GrLivArea: Above grade (ground) living area square feet
BedroomAbvGr: Number of bedrooms above grade
FullBath: Full bathrooms above grade
Target Variable:
SalePrice: The property's sale price in dollars.
Data Files:
train.csv: Contains training data with features and target variable.
To use the dataset, download the train.csv file from the Kaggle competition page and place it in the data/ directory of this repository.

Project Structure
bash
Copy code
├── data/                           # Dataset files
│   └── train.csv                   # Training dataset
├── notebooks/                      # Jupyter notebooks for exploration and model training
│   └── house_price_prediction.ipynb
├── models/                         # Saved models
│   └── linear_regression_model.pkl
├── README.md                       # Project readme file
└── requirements.txt                # Python dependencies
Usage
Load the Dataset: Ensure that the train.csv file is placed in the data/ directory.
Run the Jupyter Notebook: Open and execute the cells in notebooks/house_price_prediction.ipynb to train the model and make predictions.
Evaluate the Model: The notebook also includes code to evaluate the model's performance on the test set.
Model Evaluation
The model's performance is evaluated using the following metrics:

Mean Absolute Error (MAE): Measures the average magnitude of errors in a set of predictions, without considering their direction.
Mean Squared Error (MSE): Measures the average of the squares of the errors—that is, the average squared difference between the estimated values and what is estimated.
R-squared (R²) Score: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
Results
After training, the model's performance is as follows:

Mean Absolute Error (MAE): The average difference between the predicted and actual house prices.
Mean Squared Error (MSE): The squared average difference between the predicted and actual house prices.
R-squared (R²) Score: A score closer to 1 indicates a better fit of the model.
Visualization
A scatter plot is used to visualize the relationship between the actual and predicted house prices, helping to identify how well the model is performing.

Contributing
