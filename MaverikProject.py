#Modeling Tasks - Dom
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv("time_series_data_msba.csv")
m_xgboost = df_train.copy()
m_xgboost = m_xgboost.drop(['capital_projects.soft_opening_date', #YYYY-MM-DD
                             'calendar.calendar_day_date', #YYYY-MM-DD
                             'calendar.day_of_week', #Friday, Wednesday, etc
                            'calendar_information.holiday', # NONE, Independence Day
                             'calendar_information.type_of_day'], # WEEKDAY, WEEKEND
                            axis = 1)
m_xgboost = m_xgboost[m_xgboost['site_id_msba'] != '23065']


site_values = m_xgboost['site_id_msba'].unique()
#Unleaded train/test
unleaded = df_train.copy()
unleaded = unleaded.drop(['daily_yoy_ndt.total_inside_sales','daily_yoy_ndt.total_food_service','diesel_x'], axis = 1)
X_train_unleaded = unleaded[unleaded['site_id_msba'] != (24220,23555)]
X_test_unleaded = unleaded[unleaded['site_id_msba'] != (24220,23555)]
y_train_unleaded = unleaded[unleaded['site_id_msba'] == (24220,23555)]
y_test_unleaded = unleaded[unleaded['site_id_msba'] == (24220,23555)]

X_train_unleaded = X_train_unleaded.drop(['unleaded'], axis = 1)
y_train_unleaded = y_train_unleaded.drop(['unleaded'], axis = 1)

X_test_unleaded = X_test_unleaded['unleaded']
y_test_unleaded = y_test_unleaded['unleaded']

print(y_test_unleaded.head())
#Unleaded XGB Model
params = {'max_depth': 6,'booster': 'gbtree','eta': 0.3,'objective': 'reg:linear'}
dtrain_unleaded = xgb.DMatrix(X_train_unleaded, y_train_unleaded)
dtest_unleaded = xgb.DMatrix(X_test_unleaded, y_test_unleaded)
watchlist = [(dtrain_unleaded, 'train'), (dtest_unleaded, 'eval')]

xgboost = xgb.train(params, dtrain_unleaded, num_boost_round = 100, evals = watchlist, early_stopping_rounds = 100, verbose_eval = True)
preds = xgboost.predict(dtest_unleaded)
#RMSE of XGB model
rms_xgboost = sqrt(mean_squared_error(y_test_unleaded, preds))
print("Root Mean Squared Error for XGBoost unleaded:", rms_xgboost)
#MAE of XGB model
mae_xgboost = sqrt(mean_absolute_error(y_test_unleaded, preds))
print("Mean Absolute Error for XGBoost unleaded:", mae_xgboost)
#MAPE of XGB model
mape_xgboost = sqrt(mean_absolute_percentage_error(y_test_unleaded, preds))
print("Mean Absolute Percentage Error for XGBoost unleaded:", mape_xgboost)

#Inside XGB Model
df_train = pd.read_csv("time_series_data_msba.csv")
inside_xgboost = df_train.copy()
inside_xgboost = inside_xgboost.drop(['capital_projects.soft_opening_date', 'calendar.calendar_day_date', 'calendar.day_of_week', 'calendar_information.holiday', 'calendar_information.type_of_day'], axis=1)
#splitting the data
features = inside_xgboost.drop(["daily_yoy_ndt.total_inside_sales"], axis = 1)
target = inside_xgboost["daily_yoy_ndt.total_inside_sales"]
X_train_inside, X_test_inside, y_train_inside, y_test_inside = model_selection.train_test_split(features, target, test_size = 0.20)

params = {'max_depth': 6,'booster': 'gbtree','eta': 0.3,'objective': 'reg:linear'}
dtrain_inside = xgb.DMatrix(X_train_inside, y_train_inside)
dtest_inside = xgb.DMatrix(X_test_inside, y_test_inside)
watchlist = [(dtrain_inside, 'train'), (dtest_inside, 'eval')]

xgboost = xgb.train(params, dtrain_inside, num_boost_round = 100, evals = watchlist, early_stopping_rounds = 100, verbose_eval = True)
preds = xgboost.predict(dtest_inside)

#RMSE of XGB model
rms_xgboost_inside = sqrt(mean_squared_error(y_test_inside, preds))
print("Root Mean Squared Error for XGBoost inside:", rms_xgboost_inside)
#MAE of XGB model
mae_xgboost_inside = sqrt(mean_absolute_error(y_test_inside, preds))
print("Mean Absolute Error for XGBoost inside:", mae_xgboost_inside)
#MAPE of XGB model
mape_xgboost_inside = sqrt(mean_absolute_percentage_error(y_test_inside, preds))
print("Mean Absolute Percentage Error for XGBoost inside:", mape_xgboost_inside)

### Diesel test/train
X_train_diesel = diesel[diesel['site_id_msba'] != (24220,23555)]
X_test_diesel = diesel[diesel['site_id_msba'] != (24220,23555)]
y_train_diesel = diesel[diesel['site_id_msba'] == (24220,23555)]
y_test_diesel = diesel[diesel['site_id_msba'] == (24220,23555)]

X_train_diesel = X_train_diesel.drop(['diesel_x'], axis = 1)
y_train_diesel = y_train_diesel.drop(['diesel_x'], axis = 1)

X_test_diesel = X_test_diesel['diesel_x']
y_test_diesel = y_test_diesel['diesel_x']

print(y_test_diesel.head())
#Diesel XGB Model
diesel = m_xgboost.copy()
diesel = m_xgboost.drop(['daily_yoy_ndt.total_inside_sales','daily_yoy_ndt.total_food_service','unleaded'], axis = 1)
#splitting the data
features = diesel_xgboost.drop(["diesel_x"], axis = 1)
target = diesel_xgboost["diesel_x"]
X_train_diesel, X_test_diesel, y_train_diesel, y_test_diesel = model_selection.train_test_split(features, target, test_size = 0.20)

params = {'max_depth': 6,'booster': 'gbtree','eta': 0.3,'objective': 'reg:linear'}
dtrain_diesel = xgb.DMatrix(X_train_diesel, y_train_diesel)
dtest_diesel = xgb.DMatrix(X_test_diesel, y_test_diesel)
watchlist = [(dtrain_diesel, 'train'), (dtest_diesel, 'eval')]

xgboost = xgb.train(params, dtrain_diesel, num_boost_round = 100, evals = watchlist, early_stopping_rounds = 100, verbose_eval = True)
preds = xgboost.predict(dtest_diesel)
#RMSE of XGB model
rms_xgbrms_xgboost_diesel = sqrt(mean_squared_error(y_test_diesel, preds))
print("Root Mean Squared Error for XGBoost diesel:", rms_xgbrms_xgboost_diesel)
#MAE of XGB model
mae_xgboost_diesel = sqrt(mean_absolute_error(y_test_diesel, preds))
print("Mean Absolute Error for XGBoost diesel:", mae_xgboost_diesel)
#MAPE of XGB model
mape_xgboost_diesel = sqrt(mean_absolute_percentage_error(y_test_diesel, preds))
print("Mean Absolute Percentage Error for XGBoost diesel:", mape_xgboost_diesel)

#plotting feature importance
fig, ax = plt.subplots(figsize=(10,10))
xgb.plot_importance(xgboost_2, max_num_features=50, height=0.8, ax=ax)
plt.show()