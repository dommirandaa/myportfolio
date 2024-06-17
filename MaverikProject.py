# 1. Introduction
# Project Goal and purpose of the EDA notebook
# Maverik has tasked our group with a revenue forecasting problem. We will need to forecast the sales revenue of a store, across numerous types of items such as gas and food, day-over-day for a full year. The purpose of this EDA notebook is to explore the data that Maverik has provided that we will use to create this forecasting model.

# Business Problem
# As Maverik has a business plan of opening 30 new stores every year, the projected performance of new stores is crucial to successful selection, opening, and continual operations of these new stores while also maintaining consistent operations and growth at existing locations.

# Analytical Problem
# Our team must create a model that outperforms the current “naive” model that has been running for several weeks. The team will compare our model vs the naive model by [WHAT METRIC OF SUCCESS]. Additionally, it will forecast the daily sales of specific items such as gasoline, diesel, in-store merchandise, in-store food, and various other items. With accurate predictions, Maverik will be able to create a more accurate financial plan and provide more useful initial ROI estimations. The model performance will be measured by looking at the actual sales of each store on a given day compared to what the model predicted by [WHAT METRIC OF SUCCESS].

#EDA Tasks
## Import Data
time_data = pd.read_csv("time_series_data_msba.csv") qual_data = pd.read_csv("qualitative_data_msba.csv")
time_data.head(5)
qual_data.head(5)
#Description of the Data
# In the time series dataset, It is arranged based on dates, showcasing daily sales figures that encompass food service sales, diesel fuel sales, and unleaded fuel sales. Furthermore, it includes data that separates sales made indoors from those related to food services, and it also covers sales of both diesel and unleaded fuels. This dataset is well-suited for analyzing how sales and fuel consumption evolve over time, allowing us to uncover trends, patterns, and potential seasonal variations in these metrics. It provides valuable insights into how these sales components change from day to day.
# In the qualitative CSV, the data includes information like when the stores were opened, how big they are, how many parking spaces they have, and whether they offer certain services like specific features like pizza, Freals, or Bonfire Grill, lottery, freals, and more, indicated by "yes" or "no" values.
# Additionally, it describes the layout and features of the fueling areas, including the availability of RV lanes, high flow lanes, and services like car wash, electric vehicle charging, and propane sales. It also notes whether there are restroom facilities for both men and women. Similarly , there are details about the distance in miles to the store's location. This dataset helps us grasp the distinctive traits and attributes of different projects, all without using numerical values.

#Missing Data & Cleaning
# Describe the scope of missing data and your proposed solution
# The scope of the missing data includes columns such as traditional forecourt type, HI flow RV lanes, and RV lanes have missing values. The proposed solution is to fix the missing values is to fill the NaN with most frequent (mode) values.
# Import & Seeding
warnings.filterwarnings('ignore') np.random.seed(7)
## Data Preprocessing
time_data = time_data.sort_values(by='calendar.calendar_day_date') qual_data = qual_data.sort_values(by='open_year')
## Missing Data
print(time_data.isna().sum()) print(qual_data.isna().sum())
## Fix Missing Values
qual_data = qual_data.fillna(qual_data.mode().iloc[0])
print(qual_data.isnull().sum())
print(qual_data)

#Visual Exploration
time_data = pd.DataFrame(pd.read_csv("time_series_data_msba.csv")) qual_data = pd.DataFrame(pd.read_csv("qualitative_data_msba.csv"))
## Review indoor sales & prep
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
average_weekly_sales_indoor1 = time_data.groupby("calendar.day_of_week")["daily_yoy_ndt.total_inside_sales"].mean().loc[day_order].reset_index()
## Review food sales & prep
average_weekly_sales_food1 = time_data.groupby("calendar.day_of_week")["daily_yoy_ndt.total_food_service"].mean().loc[day_order].reset_index()
## Review diesel sales & prep
average_weekly_sales_diesel1 = time_data.groupby("calendar.day_of_week")["diesel"].mean().loc[day_order].reset_index()
## Review Unleaded sales data prep
average_weekly_sales_unleaded1 = time_data.groupby("calendar.day_of_week")["unleaded"].mean().loc[day_order].reset_index()
## Transforming indoor sales data
average_daily_sales_indoor = time_data.groupby("calendar.calendar_day_date")["daily_yoy_ndt.total_inside_sales"].mean().reset_index().sort_values("calendar.calendar_day_dat average_daily_sales_indoor['calendar.calendar_day_date'] = pd.to_datetime(average_daily_sales_indoor['calendar.calendar_day_date'], infer_datetime_format=True, errors='coer average_daily_sales_indoor1 = average_daily_sales_indoor.set_index('calendar.calendar_day_date') #Setting the Date as Index average_daily_sales_indoor1.sort_index(inplace=True)
## Transforming food sales data
average_daily_sales_food = time_data.groupby("calendar.calendar_day_date")["daily_yoy_ndt.total_food_service"].mean().reset_index().sort_values("calendar.calendar_day_date" average_daily_sales_food['calendar.calendar_day_date'] = pd.to_datetime(average_daily_sales_food['calendar.calendar_day_date'], infer_datetime_format=True, errors='coerce') average_daily_sales_food1 = average_daily_sales_food.set_index('calendar.calendar_day_date') #Setting the Date as Index average_daily_sales_food1.sort_index(inplace=True)
## Transforming diesel sales data
average_daily_sales_diesel = time_data.groupby("calendar.calendar_day_date")["diesel"].mean().reset_index().sort_values("calendar.calendar_day_date", ascending = False) average_daily_sales_diesel['calendar.calendar_day_date'] = pd.to_datetime(average_daily_sales_diesel['calendar.calendar_day_date'], infer_datetime_format=True, errors='coer average_daily_sales_diesel1 = average_daily_sales_diesel.set_index('calendar.calendar_day_date') #Setting the Date as Index average_daily_sales_diesel1.sort_index(inplace=True)
## Transforming unleaded sales data
average_daily_sales_unleaded = time_data.groupby("calendar.calendar_day_date")["unleaded"].mean().reset_index().sort_values("calendar.calendar_day_date", ascending = False) average_daily_sales_unleaded['calendar.calendar_day_date'] = pd.to_datetime(average_daily_sales_unleaded['calendar.calendar_day_date'], infer_datetime_format=True, errors=' average_daily_sales_unleaded1 = average_daily_sales_unleaded.set_index('calendar.calendar_day_date') #Setting the Date as Index average_daily_sales_unleaded1.sort_index(inplace=True)
## Weekly Seasonality Plot
x = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] indoor = average_weekly_sales_indoor1.iloc[:,1]
food = average_weekly_sales_food1.iloc[:,1]
diesel = average_weekly_sales_diesel1.iloc[:,1]
unleaded = average_weekly_sales_unleaded1.iloc[:,1]
plt.plot(x, indoor, label ='indoor', marker = 'o') plt.plot(x, food, label ='food', marker = 'o') plt.plot(x, diesel, label ='diesel', marker = 'o') plt.plot(x, unleaded, label ='unleaded', marker = 'o')
plt.xlabel("Day of Week")
plt.ylabel("Amount ($)") plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') plt.title('Weekly Average Sales by Type and Day') plt.show()

#Annual Visuals
## Indoor Sales Plot
ax = average_daily_sales_indoor1.plot(figsize=(15, 5)) ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)) ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=12)
plt.ylabel('Amount ($)', fontsize=12)
plt.title(" Average Daily Sales of Indoor", fontsize=15) plt.legend().get_texts()[0].set_text('Indoor') plt.show()
## Food Sales Plot
ax = average_daily_sales_food1.plot(figsize=(15, 5)) ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)) ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=12)
plt.ylabel('Amount ($)', fontsize=12)
plt.title(" Average Daily Sales of Food", fontsize=15) plt.legend().get_texts()[0].set_text('Food') plt.show()
## Diesel Sales Plot
ax = average_daily_sales_diesel1.plot(figsize=(15, 5)) ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)) ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=12)
plt.ylabel('Amount ($)', fontsize=12)
plt.title(" Average Daily Sales of Diesel", fontsize=15) plt.legend().get_texts()[0].set_text('Diesel') plt.show()
## Unleaded Sales Plot
#Diesel sales all time averaged plot
ax = average_daily_sales_unleaded1.plot(figsize=(15, 5)) ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)) ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=12)
plt.ylabel('Amount ($)', fontsize=12)
plt.title(" Average Daily Sales of Unleaded", fontsize=15) plt.legend().get_texts()[0].set_text('Unleaded')
plt.show()

#Modeling Tasks
import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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