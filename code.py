# --------------
# Import Libraries
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Code starts here
df=pd.read_csv(path)
df.head(5)
df.columns=df.columns.str.strip().str.lower().str.replace(' ', '_')
df.replace('NaN',np.nan)
# Code ends here


# --------------
from sklearn.model_selection import train_test_split
df.set_index(keys='serial_number',inplace=True,drop=True)


# Code starts
df['established_date']=pd.to_datetime(df['established_date'])
df['acquired_date']=pd.to_datetime(df['acquired_date'])
y=df['2016_deposits']
X=df.iloc[:,:-1]
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.25,random_state=3)
# Code ends here


# --------------
# time_col = X_train.select_dtypes(exclude=[np.number,'O']).columns
time_col = ['established_date', 'acquired_date']

# Code starts from here
for col_name in time_col:
    new_col_name = "since_"+col_name
    X_train[new_col_name] = pd.datetime.now() - X_train[col_name]
    X_train[new_col_name] = X_train[new_col_name].apply(lambda x: float(x.days)/365)
    X_train.drop(columns=col_name,inplace=True)
    
    X_val[new_col_name] = pd.datetime.now() - X_val[col_name]
    X_val[new_col_name] = X_val[new_col_name].apply(lambda x: float(x.days)/365)
    X_val.drop(columns=col_name,inplace=True)


# --------------
from sklearn.preprocessing import LabelEncoder
cat = X_train.select_dtypes(include='O').columns.tolist()

# Code starts here
X_train.fillna(value=0,axis=1,inplace=True)
X_val.fillna(value=0,axis=1,inplace=True)
le=LabelEncoder()
for i in cat:
    le.fit_transform(X_train[i])
    le.fit_transform(X_val[i])
X_train_temp=pd.get_dummies(X_train)
X_val_temp=pd.get_dummies(X_val)
# Code ends here


# --------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Code starts here
dt=DecisionTreeRegressor(random_state=5)
dt.fit(X_train,y_train)
y_pred=dt.predict(X_val)
accuracy=dt.score(X_val,y_val)
rmse=mean_squared_error(y_pred,y_val)**0.5


# --------------
from xgboost import XGBRegressor


# Code starts here
xgb=XGBRegressor(max_depth=50,learning_rate=0.83,n_estimators=100)
xgb.fit(X_train,y_train)
accuracy=xgb.score(X_val,y_val)
y_pred=xgb.predict(X_val)
rmse=mean_squared_error(y_pred,y_val)**0.5

# Code ends here


