import pandas as pd
import numpy as np
import os
import pandas as pd
from glob import glob


def get_csv_files(keyword='_all_person'):
    return [file for file in glob('*.csv') if keyword in file]


def get_single_motorcycle_crashes(df):
    '''
    Takes in a pandas dataframe that needs to be filtered down to only single motorcycle crash incidents.
    This will only work if you have an existing 'Crash ID' column.

    INPUT:
    df = Pandas dataframe with data that needs to be filtered down into only crashes with a single motorcycle

    OUTPUT:
    new_df = Filtered dataframe with only single motorcycle crashes
    '''
    original_df = df
    original_df = original_df.reset_index()
    count_of_people_involved_in_crash = original_df['crash_id'].value_counts()
    crashes_with_only_one_person = count_of_people_involved_in_crash[count_of_people_involved_in_crash == 1].index
    crashes_with_only_one_person = crashes_with_only_one_person.to_list()
    new_df = original_df[original_df['crash_id'].isin(crashes_with_only_one_person)]
    return new_df



def process_csv_files(csv_files):
    # Read the first CSV file to initialize the dataframe with columns
    df = pd.read_csv(csv_files[0])

    # Iterate over the remaining CSV files and append them to the dataframe
    for file in csv_files[1:]:
        df = pd.concat([df, pd.read_csv(file)], ignore_index=True)

    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')

    # Process the DataFrame for single motorcycle crashes
    df_svc = get_single_motorcycle_crashes(df)

    # Standardize the text in the DataFrame
    df_svc = df_svc.applymap(lambda x: x.lower().strip() if isinstance(x, str) else x)
    # Replace 'no data' with an empty string
    df_svc.replace('no data', '', inplace=True)

    return df_svc