import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing 
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('gameLogs.csv').drop(['Unnamed: 0'], axis=1)
data = data.dropna()

# 1. Check column names
print(data.columns)

# 2. Check data type of TEAM_ID column
print(data['TEAM_ID'].dtype)

# 3. Check for missing values
print(data['TEAM_ID'].isnull().sum())

def get_past_10_games(df):
    return df.groupby(['TEAM_ID', 'SEASON'], as_index=False).apply(lambda x: x.sort_values(by='GAME_DATE', ascending=False).head(10))

def feature_engineering(df):
    # Calculate total wins in the past 10 games for each team in each season
    df['TOTAL_WINS_PAST_10'] = df.groupby(['TEAM_ID', 'SEASON'])['W'].transform('sum')
    return df

# Get the past 10 games for each team in each season
past_10_games = get_past_10_games(data)

# Perform feature engineering
past_10_games = feature_engineering(past_10_games)

# Print the data types of the 'TEAM_ID' column
print(past_10_games['TEAM_ID'].dtype)


past_10_games = feature_engineering(past_10_games)

# Merge new features with existing dataset
data = pd.merge(data, past_10_games, on=['TEAM_ID', 'SEASON'])

# Train-test split
validation = data.drop(['SEASON', 'W_HOME'], axis=1)  # Features
model_data = data.drop(['SEASON', 'W_HOME'], axis=1)  # Features
target = data['W_HOME']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(model_data, target, test_size=.33)

# Standard Scaling
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Cross-validation
F1Score = cross_val_score(model, X_test_scaled, y_test, cv=12, scoring='f1_macro')
print("Logistic Model F1 Accuracy: %0.2f (+/- %0.2f)" % (F1Score.mean(), F1Score.std() * 2))

# Test Set Review
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Validation Set review
# Standard Scaling Prediction Variables
scaled_val_data = scaler.transform(validation.drop(['HOME_W','SEASON'], axis=1))

# How the model performs on unseen data
y_pred_val = model.predict(scaled_val_data)
print(classification_report(validation['HOME_W'], y_pred_val))
