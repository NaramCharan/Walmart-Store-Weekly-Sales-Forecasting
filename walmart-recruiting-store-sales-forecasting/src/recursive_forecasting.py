
import pandas as pd
class Recursive_Forecasting_:
    def __init__(self):
        pass

    def Recursive_Forecasting(self, training_set, x_val, model):
        combined_df = pd.concat(objs=(training_set, x_val), axis=0)
        combined_df = combined_df.sort_values(by=['Store', 'Dept', 'Date'])
        combined_df.reset_index(drop=True, inplace=True)
        staring_date = x_val['Date'].iloc[0]
        ending_date = combined_df['Date'].iloc[-1]
        no_of_days = (ending_date - staring_date).days
        no_of_weeks = no_of_days // 7
        date = staring_date
        for i in range(0, no_of_weeks + 1):
            combined_df.loc[combined_df['Date'] == date, 'lag_1'] = (
            combined_df.groupby(['Store', 'Dept'])['Weekly_Sales']).shift(1)
            combined_df.loc[combined_df['Date'] == date, 'lag_2'] = (
            combined_df.groupby(['Store', 'Dept'])['Weekly_Sales']).shift(2)
            combined_df.loc[combined_df['Date'] == date, 'lag_4'] = (
            combined_df.groupby(['Store', 'Dept'])['Weekly_Sales']).shift(4)
            combined_df.loc[combined_df['Date'] == date, 'rolling_4_mean'] = (
            combined_df.groupby(['Store', 'Dept'])['Weekly_Sales']).transform(
                lambda x: x.shift(1).rolling(window=4, min_periods=2).mean())
            combined_df.loc[combined_df['Date'] == date, 'rolling_8_mean'] = (
            combined_df.groupby(['Store', 'Dept'])['Weekly_Sales']).transform(
                lambda x: x.shift(1).rolling(window=8, min_periods=4).mean())
            combined_df.loc[combined_df['Date'] == date, 'rolling_4_std'] = (
            combined_df.groupby(['Store', 'Dept'])['Weekly_Sales']).transform(
                lambda x: x.shift(1).rolling(window=4, min_periods=2).std())
            combined_df.loc[combined_df['Date'] == date, 'rolling_8_std'] = (
            combined_df.groupby(['Store', 'Dept'])['Weekly_Sales']).transform(
                lambda x: x.shift(1).rolling(window=8, min_periods=4).std())
            prediction = combined_df.loc[combined_df['Date'] == date].copy()
            data_for_date = model.predict(prediction)
            prediction['Weekly_Sales'] = data_for_date
            combined_df.loc[combined_df['Date'] == date, 'Weekly_Sales'] = data_for_date
            date = date + pd.Timedelta(days=7)
        return combined_df