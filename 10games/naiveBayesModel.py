import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.metrics import classification_report

# Read the dataset
data = pd.read_csv('nbaHomeWinLossModelDataset.csv').drop(['Unnamed: 0'], axis=1)

# Drop rows with missing values
data = data.dropna()

# Filter data for training and validation
validation_seasons = ['2022-23', '2023-24', '2024-25']  # Update with your validation seasons
validation = data[data['SEASON'].isin(validation_seasons)]
modelData = data[~data['SEASON'].isin(validation_seasons)].sample(frac=1)

# Separate features and target
X = modelData.drop(['HOME_W', 'SEASON'], axis=1)
y = modelData['HOME_W']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Standard Scaling Prediction Variables
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
scaled_data_train = scaler.transform(X_train)
scaled_data_test = scaler.transform(X_test)

# Naive Bayes Model
model = GaussianNB()
model.fit(scaled_data_train, y_train)

# Model evaluation on the test set
print("Test Set Evaluation:")
print("Accuracy:", model.score(scaled_data_test, y_test))

# Cross-validation score
F1Score = cross_val_score(model, scaled_data_test, y_test, cv=12, scoring='f1_macro')
print("Naive Bayes Model F1 Accuracy: %0.2f (+/- %0.2f)" % (F1Score.mean(), F1Score.std() * 2))

# Test Set Review
y_pred = model.predict(scaled_data_test)
print("Classification Report (Test Set):")
print(classification_report(y_test, y_pred))

# Validation Set review
scaled_val_data = scaler.transform(validation.drop(['HOME_W', 'SEASON'], axis=1))
y_pred_val = model.predict(scaled_val_data)
print("Classification Report (Validation Set):")
print(classification_report(validation['HOME_W'], y_pred_val))
