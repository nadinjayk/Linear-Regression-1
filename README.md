# Simple Linear Regression 

This project implements simple linear regression in Python without using machine learning libraries like scikit-learn.

It trains a model on housing data to learn the relationship between **area** and **price** using **gradient descent**, then evaluates the model on a train-test split.

The purpose was simply to implement the gradient descent algorithm for better understanding. It results in a weak correlation as only area was used to generalise predictions for price. Can be used with other data as well.

## Features

- data preprocessing with pandas
- simple linear regression implemented from scratch
- gradient descent for parameter updates
- 80/20 train-test split
- evaluation using MSE, RMSE, MAE, and R²
- regression line visualization with matplotlib
- price prediction for a given area

## Tech Stack

- Python
- pandas
- matplotlib

## Goal

The purpose of this project is to understand how linear regression and gradient descent work internally by building them manually.

## Run

Make sure `Housing.csv` is in the same folder as the script, then run:

```bash
python linear_regression1.py
