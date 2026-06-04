
import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from src.feature_creation import MakeFeatures
from src.recursive_forecasting import Recursive_Forecasting_

features_csv = pd.read_csv('Data/features.csv')
training_csv = pd.read_csv('Data/train.csv')
testing_csv = pd.read_csv('Data/test.csv')
stores_csv = pd.read_csv('Data/stores.csv')
make_features = MakeFeatures()

df = make_features.merging_data(df=training_csv, features_csv=features_csv, stores_csv=stores_csv)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = make_features.feature_extraction(df=df)
df = df.sort_values(by=['Store', 'Dept', 'Date'], ascending=True)
df.reset_index(drop=True, inplace=True)
df['MarkDown1'] = df['MarkDown1'].fillna(0)
df['MarkDown2'] = df['MarkDown2'].fillna(0)
df['MarkDown3'] = df['MarkDown3'].fillna(0)
df['MarkDown4'] = df['MarkDown4'].fillna(0)
df['MarkDown5'] = df['MarkDown5'].fillna(0)

training_set = df[df['Date']<'2012-05-25'].copy()
validation_set = df[df['Date']>='2012-05-25'].copy()

training_set = make_features.get_time_series_features(df=training_set)

X_train = training_set.drop(columns=['Date', 'Weekly_Sales'])
y_train = training_set['Weekly_Sales']
x_val = validation_set.drop(columns=['Weekly_Sales'])
y_val = validation_set['Weekly_Sales']

#OnehotEncoding
onehotencoding_cols = ['Type', 'IsHoliday']
onehotencoding_pipeline = Pipeline(steps=[
    ("onehotencoder", OneHotEncoder(sparse_output=False))
])
#Preprocessing Pipeline
preprocessing = ColumnTransformer(transformers=[
    ('onehotencoding_pipeline', onehotencoding_pipeline, onehotencoding_cols),

], n_jobs=-1, remainder='passthrough')

#-------------------XGBoost Model -------------------
xg_model = Pipeline(steps=[
    ("preprocessing", preprocessing),
    ("xg_model", XGBRegressor(n_jobs=-1, n_estimators=1000, max_depth=14, learning_rate=0.1))])
xg_model.fit(X_train, y_train)
print("Training completed for XGBoost model")

combined_df = pd.concat(objs=(training_set, x_val), axis=0)
combined_df = combined_df.sort_values(by=['Store', 'Dept', 'Date'])
combined_df.reset_index(drop=True, inplace=True)

recursive_forecasting = Recursive_Forecasting_()

df_Xgb = recursive_forecasting.Recursive_Forecasting(training_set=training_set, x_val=x_val, model=xg_model)

xg_mse = mean_squared_error(y_val, df_Xgb[df_Xgb['Date'] >= '2012-05-25']['Weekly_Sales'])
xg_rmse = root_mean_squared_error(y_val, df_Xgb[df_Xgb['Date'] >= '2012-05-25']['Weekly_Sales'])
xg_r2score = r2_score(y_val, df_Xgb[df_Xgb['Date'] >= '2012-05-25']['Weekly_Sales'])
print(f'The Mean squared error of the xgboost model is {xg_mse}')
print(f'The Root mean squared error of the xgboost model is {xg_rmse}')
print(f'The r2 score of the xgboost model is {xg_r2score}')


#-------------------LightGBM Model -------------------
light_model = Pipeline(steps=[
    ('preprocessing', preprocessing),
    ('Light_model', LGBMRegressor(n_estimators=1000, n_jobs=-1, num_leaves=70, max_depth=16, learning_rate=0.2))
])
light_model.fit(X_train, y_train)
df_lightgbm = recursive_forecasting.Recursive_Forecasting(training_set=training_set, x_val=x_val, model=light_model)


light_mse = mean_squared_error(y_val, df_lightgbm[df_lightgbm['Date'] >= '2012-05-25']['Weekly_Sales'])
light_rmse = root_mean_squared_error(y_val, df_lightgbm[df_lightgbm['Date'] >= '2012-05-25']['Weekly_Sales'])
light_r2score = r2_score(y_val, df_lightgbm[df_lightgbm['Date'] >= '2012-05-25']['Weekly_Sales'])
print(f'The Mean squared error of the LightGBM model is {light_mse}')
print(f'The Root mean squared error of the LightGBM model is {light_rmse}')
print(f'The r2 score of the LightGBM model is {light_r2score}')
df_lightgbm.isna().sum()



#Random Forest Model
rf_model = Pipeline(steps=[
    ('preprocessing', preprocessing),
    ('rf_model', RandomForestRegressor(n_estimators=1000, n_jobs=-1, max_depth=12, min_samples_split=2, min_samples_leaf=1))
])
rf_model.fit(X_train, y_train)
df_rf = recursive_forecasting.Recursive_Forecasting(training_set=training_set, x_val=x_val,  model=rf_model)
rf_mse = mean_squared_error(y_val, df_rf[df_rf['Date'] >= '2012-05-25']['Weekly_Sales'])
rf_rmse = root_mean_squared_error(y_val, df_rf[df_rf['Date'] >= '2012-05-25']['Weekly_Sales'])
rf_r2score = r2_score(y_val, df_rf[df_rf['Date'] >= '2012-05-25']['Weekly_Sales'])

print(f'The Mean squared error of the Random Forest model is {rf_mse}')
print(f'The Root mean squared error of the Random Forest model is {rf_rmse}')
print(f'The r2 score of the Random Forest model is {rf_r2score}')


metrics_table = pd.DataFrame({
    'Models':['XGBoost_Model', 'LightGBM_Model', 'RandomForest_Model'],
    'MSE':[xg_mse, light_mse, rf_mse],
    'RMSE':[xg_rmse, light_rmse, rf_rmse],
    'R2 Score':[xg_r2score, light_r2score, rf_r2score]
})
print(metrics_table)
print("We get the LightGBM as the best model outof XGBoost , LightGBM and also RandomForest so we will now train the whole data set , using LightGBM model ")


full_data = make_features.get_time_series_features(df=df)
# data = Test_data_forecasting(training_set=df, x_val=full_data, model=light_model)
testing_data = make_features.merging_data(testing_csv, stores_csv=stores_csv, features_csv=features_csv)
testing_data['Date'] = pd.to_datetime(testing_csv['Date'], errors='coerce')
testing_data = make_features.feature_extraction(df=testing_data)
testing_data['MarkDown1'] = testing_data['MarkDown1'].fillna(0)
testing_data['MarkDown2'] = testing_data['MarkDown2'].fillna(0)
testing_data['MarkDown3'] = testing_data['MarkDown3'].fillna(0)
testing_data['MarkDown4'] = testing_data['MarkDown4'].fillna(0)
testing_data['MarkDown5'] = testing_data['MarkDown5'].fillna(0)
testing_data['CPI'] = testing_data['CPI'].ffill()
testing_data['Unemployment'] = testing_data['Unemployment'].ffill()
X_train_fulldata = full_data.drop(columns=['Date', 'Weekly_Sales'])
y_train_fulldata = full_data['Weekly_Sales']
light_model = Pipeline(steps=[
    ('preprocessing', preprocessing),
    ('Light_model', LGBMRegressor(n_estimators=1000, n_jobs=-1, num_leaves=70, max_depth=16, learning_rate=0.2))
])
light_model.fit(X_train_fulldata, y_train_fulldata)

overall_predictions = recursive_forecasting.Recursive_Forecasting(training_set=full_data, x_val=testing_data, model=light_model)
overall_predictions.to_csv('overall_predictions.csv', index=False)