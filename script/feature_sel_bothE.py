# Feature Extraction with RFE
import pandas as pd 
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas_profiling, sys, os
from matplotlib import rcParams
import matplotlib.pyplot as plt

### Run python3 feature_sel_bothE.py 2 ###

### provide energy deviation, 2 kcal/mol as in paper, free energy of barrier and reaction in paper ###
d1 = float(sys.argv[1])
ts_e = 12.3
rx_e = -4.9

### readin data file ###
df = pd.read_excel('SI_Dataset.xlsx',index_col=0)
energy = df[[' TS_deltaG',' RXN_deltaG']]

res_info = df[[38,39,40,41,42,46,64,66,67,68,70,71,72,89,90,91,92,95,117,118,119,120,139,141,142,143,144,145,146,169,170,173,174,175,198,199,300,301,302,402,411,441,458]]
res = [38,39,40,41,42,46,64,66,67,68,70,71,72,89,90,91,92,95,117,118,119,120,139,141,142,143,144,145,146,169,170,173,174,175,198,199,300,301,302,402,411,441,458]

### Class determination energy deviation within these range ###
df['Class'] = np.where((df[' TS_deltaG'] < ts_e+d1) & (df[' TS_deltaG'] > ts_e-d1) & (df[' RXN_deltaG'] < rx_e+d1) & (df[' RXN_deltaG'] > rx_e-d1), True, False)
for i in res:
    df[i] = np.where(df[i] != 0, 1, 0)

X = df[res]
y = df['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.10, random_state=42)

### remove outliers with isolation forest###
#iso = IsolationForest(contamination=0.1)
#yhat = iso.fit_predict(X_train)

### remove outliers with Minimum covariance determinant ###
#ee = EllipticEnvelope(contamination=0.01)
#yhat = ee.fit_predict(X_train)

### remove outliers with local outlier factor ###
#lof = LocalOutlierFactor()
#yhat = lof.fit_predict(X_train)

### remove outliers with one-class SVM ###
ee = OneClassSVM(nu=0.01)
yhat = ee.fit_predict(X_train)

# selection all rows that are not outliers #
mask = yhat != -1
X_train, y_train = X_train[mask,:], y_train[mask]
 
### Random Forest Classifier ###
classifier = RandomForestClassifier(n_estimators=100)

### Linear Regression Classifier ###
#classifier = LinearRegression()

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test,y_pred))
#print("Mean Absolute Error:", mean_absolute_error(y_test,y_pred))

### Identify Important Features ###

feature_importances_df = pd.DataFrame({"feature": list(X.columns),"importance": classifier.feature_importances_}).sort_values("importance",ascending=False)
print(feature_importances_df)

