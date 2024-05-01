# MachineLearningSports

## Overview
Our project aims to predict the outcomes of NBA games based on various features extracted from historical game data. This project utilizes machine learning techniques to build models that forecast whether the home team will win or lose a game. The predictions are made by analyzing factors such as team performance, rest days, and offensive efficiency. This document will provide an overview of how we collected our data, features used for our predicitions, and how our model's were built and analyzed. 

## Data Collection
The data was collected by our nba.py file. The data for this project was collected using the nba_api Python package. This framework for data extraction code came from the https://github.com/swar/nba_api which is an API allowing retrieval of NBA game data from an online database to support data analysis of basketball for a variety of motivations. The API allows querying data for specific seasons, teams, and other parameters. The collected data is stored in CSV format for further analysis. The API helped create a vast data set which gave us lots of relevant metrics to extact meaningful features. The data extracted from the online database is stored in the file **gameLogs.csv**.

## Feature Engineering
In this project, several features were engineered from the raw data to improve the predictive performance of the models. These features were carefully selected based on their potential relevance to the outcome variable, which is whether the home team wins or loses a basketball game. After experimenting with a variety of features and different combinations of them, these metrics were the most significant towards created high-performance models.These features were extracted from both the home and away team and used to perform the calculations for our models.

**Last Offensive Efficiency:** This feature represents the offensive efficiency (OE) of the team in their last game. Offensive efficiency is a measure of a team's ability to score points per possession.

**Last Game Home Win Percentage:** 
This feature indicates a team's win percentage of the home team in their last game. It provides insight into team's recent performance in a games.

**Number of Rest Days:**
The number of rest days between the team's last game and the current game. This feature captures the potential fatigue or freshness of a team.

**Last Game Away Win Percentage:** 
Similar to the second feature, this represents the team's win percentage in their away last game, reflecting their performance in away games.

**Last Game Total Win Percentage:** 
The total win percentage of the team in their last game, regardless of the location. It offers a broader perspective on the team's recent success.

**Last Game Rolling Scoring Margin:**
This feature calculates the rolling average of the scoring margin (points scored minus points allowed) for the team over a specified window of games.

**Last Game Rolling Offensive Efficiency:**
Similar to the first feature, this calculates the rolling average of the offensive efficiency for the team over a specified window of games.

## Model Creation and Analysis
Several machine learning models were built using the processed data to predict home team wins. The models included logistic regression, random forest, gradient boosting classifiers, decision trees, SVM (Support Vector Machine), and Naive Bayes classifiers. The dataset was split into training and testing sets using the previous seasons data. After scaling the data set to make ensure the data is useful for our predictions, the model is built upon the training data. The model is fit on the training data before proceeding to be scored on the testing data. This evaluation is used to calculate the initial F1 Score based on the cross validation score function which is part of the model selection package of scikit. The model is used to calculate a prediction for the scaled test data. After a prediction was made, a classification report is printed out detailing the precision, recall, F1-score, and support based on the prediction made for the testing data. We proceed to perform preprocessing by scale the validation data which is the current 22-23 season. We use our model trained on the previous seasons to make a prediciton for the current season. Similarly, a classification report is printed out for the current season data. 

Our models can be found and ran in the Model Markdown files. This markdown runs the models and prints out the output in a concise manner at the end of running. Upon running the model, the final output will display the F1-score, test classification report, and validation classification report.

**Here is an example output from the Random Forest Classifier.**
```
Analysis for Random Forest Model trained on the 2020-2021 and 2021-2022 Season Data

20-22 Random Forest Model F1 Accuracy: 0.65 (+/- 0.16)
Classification Report for Testing Data From the Training Split
              precision    recall  f1-score   support

           0       0.62      0.58      0.60       154
           1       0.68      0.71      0.69       188

    accuracy                           0.65       342
   macro avg       0.65      0.65      0.65       342
weighted avg       0.65      0.65      0.65       342

Classification Report for Predictions on the 2022-2023 Season
              precision    recall  f1-score   support

           0       0.88      0.86      0.87       468
           1       0.89      0.90      0.89       566

    accuracy                           0.88      1034
   macro avg       0.88      0.88      0.88      1034
weighted avg       0.88      0.88      0.88      1034
```
