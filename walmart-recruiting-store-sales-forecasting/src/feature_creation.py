import pandas as pd

class MakeFeatures:
    def __init__(self):
        pass

    def merging_data(self, df, stores_csv, features_csv):
        merged_stores = pd.merge(left=df, right=stores_csv, left_on='Store', right_on='Store', how='left')
        df = pd.merge(left=merged_stores, right=features_csv, left_on=['Store', 'Date', 'IsHoliday'],
                      right_on=['Store', 'Date', 'IsHoliday'], how='left')
        return df

    def feature_extraction(self, df):
        df = df.copy()
        df['Day'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        return df

    def get_time_series_features(self, df):
        df = df.copy()
        df['lag_1'] = (df.groupby(['Store', 'Dept'])['Weekly_Sales']).shift(1)
        df['lag_2'] = (df.groupby(['Store', 'Dept'])['Weekly_Sales']).shift(2)
        df['lag_4'] = (df.groupby(['Store', 'Dept'])['Weekly_Sales']).shift(4)

        df['rolling_4_mean'] = df.groupby(['Store', 'Dept'])['lag_1'].transform(
            lambda x: x.rolling(window=4, min_periods=2).mean())
        df['rolling_8_mean'] = df.groupby(['Store', 'Dept'])['lag_1'].transform(
            lambda x: x.rolling(window=8, min_periods=4).mean())

        df['rolling_4_std'] = df.groupby(['Store', 'Dept'])['lag_1'].transform(
            lambda x: x.rolling(window=4, min_periods=2).std())
        df['rolling_8_std'] = df.groupby(['Store', 'Dept'])['lag_1'].transform(
            lambda x: x.rolling(window=8, min_periods=4).std())
        df = df.dropna()
        return df