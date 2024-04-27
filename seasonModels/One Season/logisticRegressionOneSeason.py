import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing 
from sklearn.metrics import classification_report

# Load the dataset
data1 = pd.read_csv('nbaHomeWinLossModelDataset.csv').drop(['Unnamed: 0'], axis=1)
data1 = data1.dropna()

# Separate the validation set for the '2022-23' season
validation1 = data1[data1['SEASON'] == '2022-23']

# Prepare the training data for the '2021-22' season
modelData1 = data1[data1['SEASON'] == '2021-22'].sample(frac=1) 

# Separate features (X) and target variable (y)
X1 = modelData1.drop(['HOME_W','SEASON'], axis=1)
y1 = modelData1['HOME_W']

# Split the data into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=.33)

# Standard Scaling Prediction Variables
scaler1 = preprocessing.StandardScaler()
scaler1.fit(X_train1)
scaled_data_train1 = scaler1.transform(X_train1)

scaler1.fit(X_test1)
scaled_data_test1 = scaler1.transform(X_test1)

# Logistic Regression
model1 = LogisticRegression()
model1.fit(scaled_data_train1, y_train1)
model1.score(scaled_data_test1, y_test1)

# Cross-validation
F1Score1 = cross_val_score(model1, scaled_data_test1, y_test1, cv=12, scoring='f1_macro')
print("Logistic Model F1 Accuracy: %0.2f (+/- %0.2f)" % (F1Score1.mean(), F1Score1.std() * 2))

# Test Set Review
y_pred1 = model1.predict(scaled_data_test1)
print(classification_report(y_test1, y_pred1))

# Validation Set Review
# Standard Scaling Prediction Variables
scaler1 = preprocessing.StandardScaler()
scaler1.fit(validation1.drop(['HOME_W','SEASON'], axis=1))
scaled_val_data1 = scaler1.transform(validation1.drop(['HOME_W','SEASON'], axis=1))

# Evaluate the model on the validation set
y_pred_val1 = model1.predict(scaled_val_data1)
print(classification_report(validation1['HOME_W'], y_pred_val1))