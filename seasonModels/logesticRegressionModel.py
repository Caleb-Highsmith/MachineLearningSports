import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing 
from sklearn.metrics import classification_report

data = pd.read_csv('nbaHomeWinLossModelDataset.csv').drop(['Unnamed: 0'],axis=1)
data = data.dropna()
data.head(10)

validation = data[data['SEASON'] == '2022-23']
modelData = data[data['SEASON'] != '2022-23'].sample(frac=1)
X = modelData.drop(['HOME_W','SEASON'],axis=1)
y = modelData['HOME_W']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.33)

# Standard Scaling Prediction Variables
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
scaled_data_train = scaler.transform(X_train)

scaler.fit(X_test)
scaled_data_test = scaler.transform(X_test)

#Logistic Regression

model = LogisticRegression()
model.fit(scaled_data_train,y_train)
model.score(scaled_data_test,y_test)

F1Score = cross_val_score(model,scaled_data_test,y_test,cv=12,scoring='f1_macro');
print("Logistic Model F1 Accuracy: %0.2f (+/- %0.2f)"%(F1Score.mean(), F1Score.std() *2))

# Test Set Review

y_pred = model.predict(scaled_data_test)
print(classification_report(y_test,y_pred))

#Validation Set review

# Standard Scaling Prediction Variables
scaler = preprocessing.StandardScaler()
scaler.fit(validation.drop(['HOME_W','SEASON'],axis=1))
scaled_val_data = scaler.transform(validation.drop(['HOME_W','SEASON'],axis=1))
# How the model performs on unseen data
y_pred = model.predict(scaled_val_data)
print(classification_report(validation['HOME_W'],y_pred)) 

####################################################################################

# Filter data for validation from the past two seasons
validation2 = data[(data['SEASON'] == '2020-21') | (data['SEASON'] == '2021-22')]
modelData2 = data[(data['SEASON'] != '2020-21') & (data['SEASON'] != '2021-22')].sample(frac=1)
X2 = modelData2.drop(['HOME_W','SEASON'],axis=1)
y2 = modelData2['HOME_W']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=.33)

# Standard Scaling Prediction Variables
scaler2 = preprocessing.StandardScaler()
scaler2.fit(X_train2)
scaled_data_train2 = scaler2.transform(X_train2)

scaler2.fit(X_test2)
scaled_data_test2 = scaler2.transform(X_test2)

# Logistic Regression

model2 = LogisticRegression()
model2.fit(scaled_data_train2, y_train2)
model2.score(scaled_data_test2, y_test2)

F1Score2 = cross_val_score(model2, scaled_data_test2, y_test2, cv=12, scoring='f1_macro')
print("Logistic Model F1 Accuracy: %0.2f (+/- %0.2f)" % (F1Score2.mean(), F1Score2.std() * 2))

# Test Set Review

y_pred2 = model2.predict(scaled_data_test2)
print(classification_report(y_test2, y_pred2))

# Validation Set review

# Standard Scaling Prediction Variables
scaler2.fit(validation2.drop(['HOME_W','SEASON'], axis=1))
scaled_val_data2 = scaler2.transform(validation2.drop(['HOME_W','SEASON'], axis=1))

# How the model performs on unseen data
y_pred_val2 = model2.predict(scaled_val_data2)
print(classification_report(validation2['HOME_W'], y_pred_val2))

