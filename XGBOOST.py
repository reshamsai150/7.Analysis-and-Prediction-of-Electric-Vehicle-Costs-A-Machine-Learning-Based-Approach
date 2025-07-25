import numpy as np
import pandas as pd
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "EV_cars.csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "fatihilhan/electric-vehicle-specifications-and-prices",
    file_path,
)

# Replace 'NA' and empty strings with actual NaN
df.replace(['NA', ''], np.nan, inplace=True)

# Drop rows with any NaN values
df.dropna(inplace=True)

# Display cleaned data
#print("First 5 records after cleaning:", df.head())
#print(df.columns)
#print(df.shape)
#print(df.describe)
#print(df.isnull().sum())
conversion_rate = 101.54 #(1 euro in rupee)
parameter= ["Battery","Efficiency","Fast_charge","Range","Top_speed","acceleration..0.100."]
X= df[parameter]
y= df["Price.DE."] * conversion_rate

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=432)



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_sc = sc.fit_transform(X_train)  # convert all data into float data type
X_test_sc = sc.transform(X_test)



#XGBOOST
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train_sc,y_train)
y_pred_xgb_sc = model.predict(X_test_sc)
mse = mean_squared_error(y_test, y_pred_xgb_sc)
mae = mean_absolute_error(y_test, y_pred_xgb_sc)
r2 = r2_score(y_test, y_pred_xgb_sc)
print("Mean Squared Error XGB:", mse)
print("Mean Absolute Error XGB:", mae)
print("R^2 Score XGB:", r2)

"""
# Train with Standard Scalar, fit on X_train_sc achieved by scaling
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor

abd_clf_sc = AdaBoostRegressor(DecisionTreeRegressor(criterion="squared_error", random_state=20),
    n_estimators=200,
    learning_rate=0.1,
    random_state=1
)

abd_clf_sc.fit(X_train_sc,y_train)
y_pred_ada_sc=abd_clf_sc.predict(X_test_sc)
mse_ada = mean_squared_error(y_test, y_pred_ada_sc)
mae_ada = mean_absolute_error(y_test, y_pred_ada_sc)
r2_ada = r2_score(y_test, y_pred_ada_sc)
print("Mean Squared Error ADA:", mse_ada)
print("Mean Absolute Error ADA:", mae_ada)
print("R^2 Score ADA:", r2_ada)

# train with Standard Scalar, (instead of X_train, fit on X_train_sc and X_test_sc, achieve by scaling)
rf_clf_sc=RandomForestRegressor(n_estimators=20,criterion="squared_error",random_state=5)
rf_clf_sc.fit(X_train_sc,y_train)
y_pred_rf_sc=rf_clf_sc.predict(X_test_sc)
mse_random = mean_squared_error(y_test, y_pred_rf_sc)
mae_random = mean_absolute_error(y_test, y_pred_rf_sc)
r2_random = r2_score(y_test, y_pred_rf_sc)
print("Mean Squared Error Random:", mse_random)
print("Mean Absolute Error Random:", mae_random)
print("R^2 Score Random:", r2_random)
"""
#overall we found out that XGBOOST IS BEST AMONG ALL 3

import matplotlib.pyplot as plt
importances = model.feature_importances_
feature_names = X.columns
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importances")
plt.show()

