import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing 
from sklearn.metrics import classification_report

data = pd.read_csv('nbaHomeWinLossModelDataset.csv').drop(['Unnamed: 0'],axis=1)
data = data.dropna()
data.head(10)

validation = data[data['SEASON'] == '2022-23']
modelData = data[data['SEASON'] == '2021-22'].sample(frac=1)
X = modelData.drop(['HOME_W','SEASON'],axis=1)
y = modelData['HOME_W']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.33)

# Standard Scaling Prediction Variables
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
scaled_data_train = scaler.transform(X_train)

scaler.fit(X_test)
scaled_data_test = scaler.transform(X_test)

#SVC

svc_model = SVC()
svc_model.fit(scaled_data_train,y_train)
svc_model.score(scaled_data_test,y_test)

F1Score = cross_val_score(svc_model,scaled_data_test,y_test,cv=12,scoring='f1_macro');
print("Support Vector Machine F1 Accuracy: %0.2f (+/- %0.2f)"%(F1Score.mean(), F1Score.std() *2))

# Test Set Review

y_pred = svc_model.predict(scaled_data_test)
print(classification_report(y_test,y_pred))

#Validation Set review

# Standard Scaling Prediction Variables
scaler = preprocessing.StandardScaler()
scaler.fit(validation.drop(['HOME_W','SEASON'],axis=1))
scaled_val_data = scaler.transform(validation.drop(['HOME_W','SEASON'],axis=1))
# How the model performs on unseen data
y_pred = svc_model.predict(scaled_val_data)
print(classification_report(validation['HOME_W'],y_pred))