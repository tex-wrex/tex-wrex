import os
import pandas as pd

def acquire_motocycle_data():
    if os.path.exists('motocycle_data.csv'):
        df = pd.read_csv('motocycle_data.csv')
        return df
    else:
        url ='https://docs.google.com/spreadsheets/d/1jKNSokzayWWk-1_mcd-DikV9B_POOLHfuOCzV-AkI5Y/export?format=csv'
        df = pd.read_csv(url)
        url_2 = 'https://docs.google.com/spreadsheets/d/1dkvQy61WjfB769biDp7JQPXvrEHqPuofFUIlIWVo64I/export?format=csv'
        df_1 = pd.read_csv(url_2)
        merged_df = df.merge(df_1, left_on='crash_id', right_on='crash_id', how='left')
        merged_df.to_csv('motocycle_data.csv',index=False)

        return merged_df